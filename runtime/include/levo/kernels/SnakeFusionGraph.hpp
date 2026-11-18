#pragma once
// Verified graph rewrite shared by the numerical probe and the independent App.
// The original graph and external weights are never modified.
#include "../RuntimeSupport.hpp"
#include "SnakeFusedOp.hpp"
#include <MNN_generated.h>
#include <mutex>
#include <set>

namespace LevoSnakeFused {
inline void ensureRegistered() {
    static std::once_flag once;
    static bool ready = false;
    std::call_once(once, [] {
        MNN::ScheduleConfig initialize; initialize.type=MNN_FORWARD_CPU; initialize.numThread=6;
        auto runtime=MNN::Interpreter::createRuntime({initialize});
        require(!runtime.first.empty(), "Cannot initialize pinned CPU registries");
        ready=registerKernel();
    });
    require(ready, "Extra creator already occupied; refusing override");
}
inline std::vector<float> constant(const MNN::OpT& op, const std::string& weight) {
    auto blob = op.main.AsBlob(); require(blob && blob->dataType == MNN::DataType_DT_FLOAT, "Expected real FP32 constant");
    if (!blob->float32s.empty()) return blob->float32s;
    require(blob->external.size() == 2 && blob->external[1] > 0 && blob->external[1] % 4 == 0 && blob->external[1] <= 16384, "Unbounded/missing constant");
    std::vector<float> data(blob->external[1] / 4); std::ifstream file(weight, std::ios::binary);
    file.seekg(blob->external[0]); file.read(reinterpret_cast<char*>(data.data()), blob->external[1]); require(file.good(), "Constant read failed"); peak(data); return data;
}
inline std::vector<char> rewrite(const std::string& model, Json& report) {
    std::ifstream in(model, std::ios::binary | std::ios::ate); require(bool(in), "Missing current VAE graph");
    std::vector<char> data(size_t(in.tellg())); in.seekg(0); in.read(data.data(), data.size());
    std::unique_ptr<MNN::NetT> net(MNN::GetNet(data.data())->UnPack());
    std::map<std::string, MNN::OpT*> names; std::map<int, MNN::OpT*> producers; std::map<int, int> consumers;
    for (auto& op : net->oplists) { names[op->name] = op.get(); for (int id : op->outputIndexes) producers[id] = op.get(); for (int id : op->inputIndexes) ++consumers[id]; }
    std::set<std::string> removed; Json groups = Json::array();
    for (auto& p : net->oplists) {
        auto& add = *p; const std::string suffix = "/Add_1_output_0";
        if (!LevoVaeSnake::ends(add.name, suffix)) continue;
        auto prefix = add.name.substr(0, add.name.size() - suffix.size());
        if (!names.count(prefix + "/Sin_output_0")) continue;
        auto mul1 = names.at(prefix + "/Mul_1_output_0"), sine = names.at(prefix + "/Sin_output_0"),
             power = names.at(prefix + "/Pow_output_0"), mul2 = names.at(prefix + "/Mul_2_output_0");
        require(add.main.AsBinaryOp()&&add.main.AsBinaryOp()->opType==MNN::BinaryOpOperation_ADD&&mul1->main.AsBinaryOp()&&mul1->main.AsBinaryOp()->opType==MNN::BinaryOpOperation_MUL&&
            mul2->main.AsBinaryOp()&&mul2->main.AsBinaryOp()->opType==MNN::BinaryOpOperation_MUL&&sine->main.AsUnaryOp()&&sine->main.AsUnaryOp()->opType==MNN::UnaryOpOperation_SIN,"Snake operator semantics changed");
        require(add.inputIndexes.size() == 2 && mul1->inputIndexes.size() == 2 && mul2->inputIndexes.size() == 2 && power->inputIndexes.size() >= 1,
            "Snake topology mismatch " + prefix + Json({{"add",add.inputIndexes},{"mul1",mul1->inputIndexes},{"mul2",mul2->inputIndexes},{"power",power->inputIndexes}}).dump());
        int x = -1, alphaId = -1, betaId = -1;
        for (int id : mul1->inputIndexes) { if (producers.at(id)->type == MNN::OpType_Const) alphaId = id; else x = id; }
        for (int id : mul2->inputIndexes) if (id != power->outputIndexes.at(0)) betaId = id;
        require(x >= 0 && alphaId >= 0 && betaId >= 0 && std::find(add.inputIndexes.begin(), add.inputIndexes.end(), x) != add.inputIndexes.end(), "Snake residual/constant mismatch");
        require(std::find(add.inputIndexes.begin(),add.inputIndexes.end(),mul2->outputIndexes.at(0))!=add.inputIndexes.end()&&std::find(mul2->inputIndexes.begin(),mul2->inputIndexes.end(),power->outputIndexes.at(0))!=mul2->inputIndexes.end(),"Snake tail connectivity mismatch");
        require(sine->inputIndexes == mul1->outputIndexes && power->inputIndexes[0] == sine->outputIndexes[0], "Snake sequence mismatch");
        if(power->inputIndexes.size()==2){auto exponent = constant(*producers.at(power->inputIndexes[1]), model + ".weight"); require(exponent.size() == 1 && exponent[0] == 2, "Snake power must be2");}
        else {auto unary=power->main.AsUnaryOp();require(unary&&unary->opType==MNN::UnaryOpOperation_SQUARE,"Expected optimized square");}
        for (auto op : {mul1, sine, power, mul2}) require(consumers[op->outputIndexes.at(0)] == 1, "Refuse shared intermediate");
        auto a = constant(*producers.at(alphaId), model + ".weight"), b = constant(*producers.at(betaId), model + ".weight");
        require(a.size() == b.size() && a.size() >= 8 && a.size() <= 2048 && a.size() % 8 == 0, "Snake channel count mismatch");
        MNN::ExtraT extra; extra.engine = "LevoResearch"; extra.type = "SnakeFp16V1"; extra.vector = true; extra.info.resize(a.size() * 8);
        std::memcpy(extra.info.data(), a.data(), a.size() * 4); std::memcpy(extra.info.data() + a.size() * 4, b.data(), b.size() * 4);
        for (auto op : {mul1, sine, power, mul2}) removed.insert(op->name);
        add.type = MNN::OpType_Extra; add.main.Set(std::move(extra)); add.inputIndexes = {x};
        groups.push_back({{"name", add.name}, {"channels", a.size()}});
    }
    require(groups.size() == 36, "Expected exactly36 independently verified Snake chains");
    const auto before = net->oplists.size();
    net->oplists.erase(std::remove_if(net->oplists.begin(), net->oplists.end(), [&](const std::unique_ptr<MNN::OpT>& op) {return removed.count(op->name);}), net->oplists.end());
    flatbuffers::FlatBufferBuilder builder; builder.Finish(MNN::Net::Pack(builder, net.get()));
    report = {{"groups", groups}, {"operatorsBefore", before}, {"operatorsAfter", net->oplists.size()}, {"pointwiseNodesRemoved", removed.size()},
              {"sourceWeightsUnmodified", true}, {"producerSnapshotBytes", 0}, {"logicalActivationTraffic", "Five stages11N FP16 reads/writes -> fused2N; estimate, not measured DRAM bytes"}};
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}
}
