// Export an experimental paired projection for the pinned 0.6B W4 graph.
// Keep original quant codes, scales and the existing MUL_SILU semantics.
#include <MNN_generated.h>
#include <json.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
using Json = nlohmann::json;
constexpr int Rows = 3072;
constexpr int Cols = 1024;
constexpr int Layers = 28;
constexpr size_t HeaderBytes = 22;
constexpr size_t PackedBytes = Rows * Cols / 2;
constexpr size_t AlphaBytes = Rows * (Cols / 64) * 2;
constexpr size_t PartBytes = HeaderBytes + PackedBytes + AlphaBytes;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

using Offset = uint64_t;

void copy(std::ifstream& in, std::ofstream& out, Offset begin, Offset length) {
    in.seekg(begin);
    require(bool(in), "Cannot seek source weights");
    std::array<char, 65536> buffer{};
    while (length) {
        const auto count = std::min<Offset>(length, buffer.size());
        require(bool(in.read(buffer.data(), count)), "Short source weight read");
        require(bool(out.write(buffer.data(), count)), "Output weight write failed");
        length -= count;
    }
}

void validate(const MNN::Convolution2DT& parameter) {
    const auto& spec = parameter.common;
    const auto& q = parameter.quanParameter;
    require(spec && spec->inputCount == Cols && spec->outputCount == Rows &&
        spec->kernelX == 1 && spec->kernelY == 1 && spec->strideX == 1 &&
        spec->strideY == 1 && spec->dilateX == 1 && spec->dilateY == 1 &&
        spec->group == 1 && spec->padX == 0 && spec->padY == 0 &&
        spec->pads.empty() && spec->outPads.empty() && !spec->hasOutputShape &&
        !spec->relu && !spec->relu6, "Expected plain equal 1x1 projections");
    require(q && q->type == 1 && q->aMaxOrBits == 4 && !q->shapeInt32 &&
        q->scaleStorage == MNN::ScaleStorageType_FP16 &&
        q->buffer.empty() && q->alpha.empty() && q->alphaFp16.empty() &&
        !parameter.symmetricQuan && !parameter.sparseParameter &&
        parameter.weight.empty() && parameter.bias.empty(), "Unsupported quant encoding");
    const auto& ref = parameter.external;
    require(ref.size() == 5 && ref[0] >= 0 && ref[1] == HeaderBytes + PackedBytes &&
        ref[2] == AlphaBytes && ref[3] == 0 && ref[4] == 0, "Unexpected external packing");
}

template <typename Table, typename Object>
std::vector<uint8_t> packedMetadata(const Object* object) {
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Table::Pack(builder, object));
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

struct Pair {
    Offset up;
    Offset gate;
};

std::array<uint8_t, HeaderBytes> readHeader(std::ifstream& in, Offset position) {
    in.seekg(position);
    std::array<uint8_t, HeaderBytes> header{};
    require(bool(in.read(reinterpret_cast<char*>(header.data()), header.size())), "Short quant header");
    require(header[0] == 2 && (header[1] | header[2] << 8) == Rows &&
        (header[3] | header[4] << 8) == Cols && header[5] == 16,
        "Unexpected Q4 dense shape or codebook size");
    for (int i = 0; i < 16; ++i) {
        require(header[6 + i] == static_cast<uint8_t>(i - 8), "Unexpected Q4 codebook");
    }
    return header;
}

void writePair(std::ifstream& in, std::ofstream& out, const Pair& pair) {
    auto header = readHeader(in, pair.up);
    require(header == readHeader(in, pair.gate), "Q4 codebooks differ");
    header[1] = (2 * Rows) & 255;
    header[2] = (2 * Rows) >> 8;
    out.write(reinterpret_cast<const char*>(header.data()), header.size());
    for (auto off : {pair.up, pair.gate}) copy(in, out, off + HeaderBytes, PackedBytes);
    for (auto off : {pair.up, pair.gate}) copy(in, out, off + HeaderBytes + PackedBytes, AlphaBytes);
    // Preserve every unrelated op's external offset.
    const std::array<char, HeaderBytes> padding{};
    out.write(padding.data(), padding.size());
}

void build(const std::string& source, const std::string& destination, bool directGate, Json& report) {
    require(!std::filesystem::exists(destination), "Preserve prior output");
    std::ifstream graph(source, std::ios::binary | std::ios::ate);
    require(graph && graph.tellg() > 0 && graph.tellg() < 8 * 1024 * 1024, "Missing or invalid graph");
    std::vector<uint8_t> bytes(static_cast<size_t>(graph.tellg()));
    graph.seekg(0);
    require(bool(graph.read(reinterpret_cast<char*>(bytes.data()), bytes.size())), "Short graph read");
    flatbuffers::Verifier verify(bytes.data(), bytes.size());
    require(MNN::VerifyNetBuffer(verify), "Invalid MNN graph");
    std::unique_ptr<MNN::NetT> net(MNN::GetNet(bytes.data())->UnPack());
    require(net && net->subgraphs.empty(), "Expected one static graph");
    std::unordered_map<std::string, size_t> names;
    std::set<std::string> ambiguous;
    std::unordered_map<int, size_t> uses;
    for (size_t i = 0; i < net->oplists.size(); ++i) {
        const auto& op = net->oplists[i];
        if (!names.emplace(op->name, i).second) ambiguous.insert(op->name);
        for (int input : op->inputIndexes) ++uses[input];
        for (int output : op->outputIndexes) {
            require(output >= 0 && size_t(output) < net->tensorName.size(), "Invalid graph tensor index");
        }
    }
    std::ifstream weights(source + ".weight", std::ios::binary | std::ios::ate);
    require(weights && weights.tellg() > 0, "Missing external weights");
    const auto weightBytes = static_cast<Offset>(weights.tellg());
    std::map<Offset, Pair> replacements;
    std::map<size_t, std::unique_ptr<MNN::OpT>> insert;
    std::set<size_t> remove;
    const auto lookup = [&](const std::string& name) {
        require(!ambiguous.count(name), "Ambiguous target operator name");
        return names.at(name);
    };
    for (int layer = 0; layer < Layers; ++layer) {
        const auto path = "/layers." + std::to_string(layer) + "/mlp/";
        const auto gateIndex = lookup(path + "gate_proj/Linear");
        const auto upIndex = lookup(path + "up_proj/Linear");
        const auto mulIndex = lookup("/blocks." + std::to_string(layer) + "/mlp/Mul_output_0");
        auto& up = net->oplists[upIndex];
        auto& gate = net->oplists[gateIndex];
        const auto& mul = net->oplists[mulIndex];
        const auto* binary = mul->main.AsBinaryOp();
        require(up->type == MNN::OpType_Convolution && gate->type == MNN::OpType_Convolution &&
            gateIndex < upIndex && upIndex < mulIndex && up->inputIndexes.size() == 1 &&
            up->inputIndexes == gate->inputIndexes && up->outputIndexes.size() == 1 &&
            gate->outputIndexes.size() == 1 && up->externalPath == gate->externalPath &&
            up->defaultDimentionFormat == gate->defaultDimentionFormat, "Unexpected projection dependencies");
        const int upResult = up->outputIndexes.front(), gateResult = gate->outputIndexes.front();
        require(binary && binary->opType == MNN::BinaryOpOperation_MUL_SILU &&
            mul->inputIndexes == std::vector<int>{upResult, gateResult} &&
            uses[upResult] == 1 && uses[gateResult] == 1, "Unexpected gate inputs or fan-out");
        for (auto result : {upResult, gateResult}) {
            require(std::find(net->outputName.begin(), net->outputName.end(), net->tensorName[result]) ==
                net->outputName.end(), "A projection is a graph output");
        }
        auto* u = up->main.AsConvolution2D();
        auto* g = gate->main.AsConvolution2D();
        require(u && g, "Missing projection parameters");
        validate(*u);
        validate(*g);
        require(packedMetadata<MNN::Convolution2DCommon>(u->common.get()) ==
                packedMetadata<MNN::Convolution2DCommon>(g->common.get()) &&
            packedMetadata<MNN::IDSTQuan>(u->quanParameter.get()) ==
                packedMetadata<MNN::IDSTQuan>(g->quanParameter.get()), "Projection quant or convolution metadata differs");
        const Offset start = g->external[0], right = u->external[0];
        require(start + PartBytes == right && right + PartBytes <= weightBytes, "Projection weights must be consecutive");
        require(replacements.emplace(start, Pair{right, start}).second, "Duplicated projection weight range");

        const auto pairName = path + "GateUpPairW4";
        require(!names.count(pairName) && !names.count(pairName + "/Split"), "Paired projection already exists");
        const int merged = static_cast<int>(net->tensorName.size());
        net->tensorName.push_back(pairName + ":0");
        u->common->outputCount = 2 * Rows;
        u->external = {static_cast<int64_t>(start), HeaderBytes + 2 * PackedBytes, 2 * AlphaBytes, 0, 0};
        up->name = pairName;
        up->outputIndexes = {merged};

        remove.insert(gateIndex);
        if (directGate) {
            auto fused = std::make_unique<MNN::OpT>();
            fused->type = MNN::OpType_Plugin;
            fused->name = pairName + "/DirectGate";
            fused->defaultDimentionFormat = mul->defaultDimentionFormat;
            fused->inputIndexes = {merged};
            fused->outputIndexes = mul->outputIndexes;
            MNN::PluginT plugin;
            plugin.type = "LevoPairedW4MulSilu";
            fused->main.Set(std::move(plugin));
            remove.insert(mulIndex);
            insert.emplace(mulIndex, std::move(fused));
            continue;
        }
        auto slice = std::make_unique<MNN::OpT>();
        slice->type = MNN::OpType_Slice;
        slice->name = pairName + "/Split";
        slice->defaultDimentionFormat = up->defaultDimentionFormat;
        slice->inputIndexes = {merged};
        slice->outputIndexes = {upResult, gateResult};
        MNN::SliceT split;
        split.axis = 1;
        split.slicePoints = {Rows};
        slice->main.Set(std::move(split));
        insert.emplace(upIndex, std::move(slice));
    }
    std::vector<std::unique_ptr<MNN::OpT>> operations;
    operations.reserve(net->oplists.size());
    for (size_t i = 0; i < net->oplists.size(); ++i) {
        if (!remove.count(i)) operations.push_back(std::move(net->oplists[i]));
        if (auto it = insert.find(i); it != insert.end()) operations.push_back(std::move(it->second));
    }
    net->oplists = std::move(operations);
    if (net->tensorNumber) net->tensorNumber = net->tensorName.size();
    require(std::filesystem::create_directory(destination), "Cannot create candidate directory");
    std::ofstream out(destination + "/llm.mnn.weight", std::ios::binary);
    Offset cursor = 0;
    for (const auto& [start, pair] : replacements) {
        require(start >= cursor, "Overlapping weight replacement");
        copy(weights, out, cursor, start - cursor);
        writePair(weights, out, pair);
        cursor = start + 2 * PartBytes;
    }
    copy(weights, out, cursor, weightBytes - cursor);
    out.close();
    require(out.good() && std::filesystem::file_size(destination + "/llm.mnn.weight") == weightBytes,
        "Incorrect external weights file length");
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(MNN::Net::Pack(builder, net.get()));
    std::ofstream result(destination + "/llm.mnn", std::ios::binary);
    result.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
    result.close();
    require(result.good(), "Cannot write candidate graph");
    report = {{"success", true}, {"pairedSites", Layers}, {"sourceWeightBytes", weightBytes},
        {"candidateWeightBytes", weightBytes}, {"inputWidth", Cols}, {"pairedWidth", 2 * Rows},
        {"directGate", directGate}, {"scope", directGate
            ? "Exact Q4 repack plus direct packed-input MUL_SILU. Requires the experimental runner."
            : "Exact stacking of encoded Q4 rows and FP16 scales. MNN Slice and original MUL_SILU remain."}};
}
}

int main(int argc, char** argv) {
    if (argc != 3 && !(argc == 4 && std::string(argv[3]) == "--direct-gate")) return 2;
    Json report;
    try {
        build(argv[1], argv[2], argc == 4, report);
    } catch (const std::exception& error) {
        report = {{"success", false}, {"error", error.what()}};
    }
    std::cout << report.dump() << '\n';
    return report["success"] == true ? 0 : 1;
}
