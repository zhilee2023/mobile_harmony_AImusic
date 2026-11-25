// Isolated DiT timing for one candidate produced by the operator-lab search.
#include "../runtime/include/levo/experimental/StreamLab.hpp"
#include <MNN_generated.h>
#include <array>
#include <set>

using namespace LevoStreamLab;

namespace {
constexpr int frames = 256;
constexpr int channels = 64;
constexpr int inputWidth = 1200;

float scalar(const MNN::OpT& node, std::ifstream& weights) {
    const auto* blob = node.main.AsBlob();
    require(blob && blob->dataType == MNN::DataType_DT_FLOAT, "Expected FP32 GELU coefficient");
    if (blob->float32s.size() == 1) return blob->float32s.front();
    require(blob->external.size() == 2 && blob->external[1] == sizeof(float),
            "Expected a single external GELU coefficient");
    float value = 0;
    weights.seekg(blob->external[0]);
    require(bool(weights.read(reinterpret_cast<char*>(&value), sizeof(value))), "Cannot read GELU coefficient");
    return value;
}

std::vector<char> rewrite(const std::string& graphPath, const std::string& weightsPath, bool cubeOnly) {
    std::ifstream file(graphPath, std::ios::binary | std::ios::ate);
    require(file && file.tellg() > 0 && file.tellg() < 64 * 1024 * 1024, "Invalid source graph");
    std::vector<char> bytes(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    require(bool(file.read(bytes.data(), bytes.size())), "Cannot read source graph");
    flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
    require(MNN::VerifyNetBuffer(verifier), "Cannot verify source graph");
    std::unique_ptr<MNN::NetT> net(MNN::GetNet(bytes.data())->UnPack());
    std::map<std::string, MNN::OpT*> names;
    std::map<int, MNN::OpT*> producers;
    std::map<int, int> consumers;
    for (auto& op : net->oplists) {
        require(names.emplace(op->name, op.get()).second, "Non-unique source operator name");
        for (const auto input : op->inputIndexes) ++consumers[input];
        for (const auto output : op->outputIndexes) {
            require(producers.emplace(output, op.get()).second, "Non-unique source tensor");
        }
    }
    std::ifstream weights(weightsPath, std::ios::binary);
    require(bool(weights), "Missing original weights");
    std::set<const MNN::OpT*> removed;
    std::map<const MNN::OpT*, std::unique_ptr<MNN::OpT>> insertBefore;
    for (int layer = 0; layer < 22; ++layer) {
        const auto base = "/cfm/estimator/h." + std::to_string(layer) + "/mlp/";
        const auto act = base + "act/";
        const auto get = [&](const char* suffix) { return names.at(act + suffix); };
        const auto out = [](const MNN::OpT& node) { return node.outputIndexes.at(0); };
        const auto checkBinary = [](const MNN::OpT& node, MNN::BinaryOpOperation kind) {
            const auto* params = node.main.AsBinaryOp();
            require(node.type == MNN::OpType_BinaryOp && params && params->opType == kind &&
                    params->T == MNN::DataType_DT_FLOAT && params->activationType == 0 &&
                    node.inputIndexes.size() == 2 && node.outputIndexes.size() == 1,
                    "Unexpected GELU binary operator");
        };
        const auto checkEdges = [](const MNN::OpT& node, int a, int b) {
            require((node.inputIndexes == std::vector<int>{a, b}) ||
                    (node.inputIndexes == std::vector<int>{b, a}), "GELU dependencies changed");
        };
        const auto checkCoefficient = [&](const MNN::OpT& node, int input, float expected) {
            const auto& ids = node.inputIndexes;
            require(ids[0] == input || ids[1] == input, "Missing GELU operand");
            const auto other = ids[0] == input ? ids[1] : ids[0];
            const auto* coefficient = producers.at(other);
            require(coefficient->type == MNN::OpType_Const &&
                    std::abs(scalar(*coefficient, weights) - expected) <= 1e-7f,
                    "GELU coefficient differs from native approximation");
        };
        const auto* half = get("Mul_output_0");
        auto* power = get("Pow_output_0");
        const auto* cubic = get("Mul_1_output_0");
        const auto* sum = get("Add_output_0");
        const auto* scale = get("Mul_2_output_0");
        const auto* tanh = get("Tanh_output_0");
        const auto* addOne = get("Add_1_output_0");
        auto* product = get("Mul_3_output_0");
        const auto x = out(*names.at(base + "c_fc/Gemm_output_0"));
        for (const auto* node : {half, cubic, scale, static_cast<const MNN::OpT*>(product)}) {
            checkBinary(*node, MNN::BinaryOpOperation_MUL);
        }
        checkBinary(*power, MNN::BinaryOpOperation_POW);
        checkBinary(*sum, MNN::BinaryOpOperation_ADD);
        checkBinary(*addOne, MNN::BinaryOpOperation_ADD);
        require(power->inputIndexes.front() == x, "GELU power base changed");
        checkCoefficient(*power, x, 3.f);
        checkCoefficient(*half, x, .5f);
        checkCoefficient(*cubic, out(*power), .044715f);
        checkEdges(*sum, x, out(*cubic));
        checkCoefficient(*scale, out(*sum), .79788458f);
        const auto* unary = tanh->main.AsUnaryOp();
        require(tanh->type == MNN::OpType_UnaryOp && unary && unary->opType == MNN::UnaryOpOperation_TANH &&
                unary->T == MNN::DataType_DT_FLOAT && tanh->inputIndexes == std::vector<int>{out(*scale)} &&
                tanh->outputIndexes.size() == 1, "GELU tanh changed");
        checkCoefficient(*addOne, out(*tanh), 1.f);
        checkEdges(*product, out(*half), out(*addOne));
        if (cubeOnly) {
            auto square = std::make_unique<MNN::OpT>();
            square->name = act + "PricingSquare";
            square->type = MNN::OpType_BinaryOp;
            square->defaultDimentionFormat = power->defaultDimentionFormat;
            const int tensor = static_cast<int>(net->tensorName.size());
            net->tensorName.push_back(square->name);
            square->inputIndexes = {x, x};
            square->outputIndexes = {tensor};
            MNN::BinaryOpT params;
            params.T = MNN::DataType_DT_FLOAT;
            params.opType = MNN::BinaryOpOperation_MUL;
            square->main.Set(MNN::BinaryOpT(params));
            power->inputIndexes = {tensor, x};
            power->main.Set(std::move(params));
            insertBefore.emplace(power, std::move(square));
            continue;
        }
        for (const auto* node : {half, static_cast<const MNN::OpT*>(power), cubic, sum, scale, tanh, addOne}) {
            require(consumers[out(*node)] == 1, "GELU intermediate has an external consumer");
            require(std::find(net->outputName.begin(), net->outputName.end(), net->tensorName.at(out(*node))) ==
                    net->outputName.end(), "GELU intermediate is a graph output");
            removed.insert(node);
        }
        MNN::UnaryOpT fused;
        fused.opType = MNN::UnaryOpOperation_GELU;
        fused.T = MNN::DataType_DT_FLOAT;
        product->name = act + "GeluPricing";
        product->type = MNN::OpType_UnaryOp;
        product->main.Set(std::move(fused));
        product->inputIndexes = {x};
    }
    if (cubeOnly) {
        require(insertBefore.size() == 22, "Expected 22 inserted square operators");
        std::vector<std::unique_ptr<MNN::OpT>> operators;
        operators.reserve(net->oplists.size() + insertBefore.size());
        for (auto& op : net->oplists) {
            auto insertion = insertBefore.find(op.get());
            if (insertion != insertBefore.end()) operators.push_back(std::move(insertion->second));
            operators.push_back(std::move(op));
        }
        net->oplists = std::move(operators);
        net->tensorNumber = net->tensorName.size();
    } else {
        require(removed.size() == 154, "Expected seven removed nodes in each of 22 layers");
        net->oplists.erase(std::remove_if(net->oplists.begin(), net->oplists.end(),
                          [&](const auto& op) { return removed.count(op.get()) != 0; }), net->oplists.end());
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(MNN::Net::Pack(builder, net.get()));
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

std::vector<float> conditioning(const std::string& root, const std::string& codesPath) {
    const auto codes = readJson(codesPath).get<std::vector<int>>();
    require(codes.size() >= frames, "Need at least 256 actual music codes");
    MappedFloats table(root + "/conditioner/cond-token.f16");
    std::vector<float> mu(frames * 1024);
    for (int i = 0; i < frames; ++i) {
        require(codes[i] >= 0 && codes[i] < 16384, "Invalid audio token");
        table.row(codes[i], 1024, mu.data() + i * 1024);
    }
    return mu;
}

const char* checkpoint(const std::string& name) {
    const auto first = std::string("/cfm/estimator/h.0/mlp/");
    if (name == first + "c_fc/Gemm_output_0__matmul_converted") return "up";
    if (name == first + "act/Pow_output_0") return "power";
    if (name == first + "act/Mul_3_output_0" || name == first + "act/GeluPricing") return "gelu";
    if (name == first + "c_proj/Gemm_output_0__matmul_converted") return "down";
    if (name == "/cfm/estimator/h.10/mlp/c_proj/Gemm_output_0__matmul_converted") return "down10";
    if (name == "/cfm/estimator/h.21/mlp/c_proj/Gemm_output_0__matmul_converted") return "down21";
    return nullptr;
}

void capture(const std::string& path, MNN::Tensor* tensor) {
    MNN::Tensor host(tensor, MNN::Tensor::CAFFE, true);
    require(host.getType().bits == 32 && tensor->copyToHostTensor(&host), "Checkpoint capture failed");
    std::ofstream file(path, std::ios::binary);
    require(bool(file.write(reinterpret_cast<const char*>(host.host<float>()),
                            host.elementSize() * sizeof(float))), "Checkpoint write failed");
}
}

int main(int argc, char** argv) {
    if (argc < 5 || argc > 7) {
        std::cerr << "bundleDir codes.json outputDir stock|gelu|cube [--trace] [--float-activation]\n";
        return 2;
    }
    bool traceNumerics = false, floatActivation = false;
    for (int arg = 5; arg < argc; ++arg) {
        const std::string flag = argv[arg];
        if (flag == "--trace" && !traceNumerics) traceNumerics = true;
        else if (flag == "--float-activation" && !floatActivation) floatActivation = true;
        else return 2;
    }
    const std::string bundle = argv[1], output = argv[3], mode = argv[4];
    Json report = {{"success", false}, {"mode", mode}, {"fixture", "first 256 real music codes, fixed noise"},
                   {"precision", "CPU High"}, {"threads", 6}, {"batch", 2}, {"cfgScale", 1.5},
                   {"memoryMode", floatActivation ? "normal" : "low"},
                   {"stepsPerRound", 2}, {"rounds", Json::array()}, {"checkpointTrace", traceNumerics}};
    if (std::filesystem::exists(output)) return 3;
    std::filesystem::create_directories(output);
    Trace trace;
    try {
        require(mode == "stock" || mode == "gelu" || mode == "cube", "Unknown probe mode");
        const bool rewritten = mode != "stock";
        const auto path = bundle + "/dit/dit.mnn";
        const auto weight = path + ".weight";
        const auto bytes = rewritten ? rewrite(path, weight, mode == "cube") : std::vector<char>();
        report["rewrittenSites"] = rewritten ? 22 : 0;
        auto tick = Clock::now();
        std::unique_ptr<MNN::Interpreter> net(rewritten ? MNN::Interpreter::createFromBuffer(bytes.data(), bytes.size())
                                                   : MNN::Interpreter::createFromFile(path.c_str()));
        require(bool(net), "Cannot load DiT");
        net->setExternalFile(weight.c_str());
        net->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
        if (traceNumerics) net->setSessionMode(MNN::Interpreter::Session_Debug);
        net->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI, 0);
        MNN::BackendConfig backend;
        backend.precision = MNN::BackendConfig::Precision_High;
        backend.memory = floatActivation ? MNN::BackendConfig::Memory_Normal : MNN::BackendConfig::Memory_Low;
        if (floatActivation) require(available() > 3000 * 1024L, "Need 3000 MiB available for expanded weights");
        backend.power = MNN::BackendConfig::Power_High;
        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_CPU;
        config.numThread = 6;
        config.backendConfig = &backend;
        auto* session = net->createSession(config);
        require(session, "Cannot create DiT session");
        net->resizeSession(session);
        net->releaseModel();
        report["setupSeconds"] = seconds(tick);
        const auto put = [&](const char* name, const std::vector<float>& values) {
            auto* tensor = net->getSessionInput(session, name);
            require(tensor, "Missing DiT input");
            MNN::Tensor host(tensor, MNN::Tensor::CAFFE, true);
            require(static_cast<size_t>(host.elementSize()) == values.size(), "Unexpected DiT input shape");
            std::memcpy(host.host<float>(), values.data(), values.size() * sizeof(float));
            require(tensor->copyFromHostTensor(&host), "DiT input transfer failed");
        };
        const auto mu = conditioning(bundle, argv[2]);
        const auto masks = read(bundle + "/aux/mask.f32", 144);
        std::mt19937 random(2307);
        std::normal_distribution<float> normal;
        std::vector<float> noise(frames * channels);
        for (auto& value : noise) value = normal(random);
        std::vector<float> inputs(2 * frames * inputWidth);
        for (int b = 0; b < 2; ++b) {
            for (int i = 0; i < frames; ++i) {
                auto* row = inputs.data() + (b * frames + i) * inputWidth;
                std::copy_n(masks.data() + 96, 48, row);
                if (b) std::copy_n(mu.data() + i * 1024, 1024, row + 112);
            }
        }
        const std::vector<float> attention(2 * frames * frames, 1.f);
        std::ofstream reference(output + "/velocity.f32", std::ios::binary);
        Json checkpoints = Json::array();
        for (int round = 0; round < (traceNumerics ? 1 : 3); ++round) {
            auto latent = noise;
            Json steps = Json::array();
            for (int step = 0; step < 2; ++step) {
                check();
                const auto start = Clock::now();
                for (int n = 0; n < 2 * frames; ++n) {
                    std::copy_n(latent.data() + (n % frames) * channels, channels, inputs.data() + n * inputWidth + 1136);
                }
                put("inputs", inputs);
                put("attention_mask", attention);
                put("timestep", std::vector<float>(2, step * .5f));
                const auto kernel = Clock::now();
                if (traceNumerics && step == 0) {
                    const auto before = [](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo*) { return true; };
                    const auto after = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
                        if (const auto* name = checkpoint(info->name())) {
                            require(tensors.size() == 1, "Expected one checkpoint tensor");
                            capture(output + "/" + name + ".f32", tensors.front());
                            checkpoints.push_back({{"name", name}, {"shape", tensors.front()->shape()}});
                        }
                        return true;
                    };
                    require(net->runSessionWithCallBackInfo(session, before, after, true) == MNN::NO_ERROR,
                            "DiT checkpoint run failed");
                } else {
                    require(net->runSession(session) == MNN::NO_ERROR, "DiT forward failed");
                }
                const double forward = seconds(kernel);
                auto* result = net->getSessionOutput(session, "velocity");
                require(result, "Missing DiT velocity");
                MNN::Tensor host(result, MNN::Tensor::CAFFE, true);
                require(host.elementSize() == 2 * frames * channels && result->copyToHostTensor(&host),
                        "DiT velocity transfer failed");
                const double total = seconds(start);
                const auto* velocity = host.host<float>();
                for (int i = 0; i < host.elementSize(); ++i) require(std::isfinite(velocity[i]), "Non-finite velocity");
                for (size_t i = 0; i < latent.size(); ++i) {
                    const float unconditioned = velocity[i];
                    const float conditioned = velocity[i + latent.size()];
                    latent[i] += .5f * (unconditioned + 1.5f * (conditioned - unconditioned));
                }
                if (round == 0) {
                    reference.write(reinterpret_cast<const char*>(velocity), host.elementSize() * sizeof(float));
                    require(bool(reference), "Cannot save paired velocity");
                }
                steps.push_back({{"forwardSeconds", forward}, {"totalSeconds", total}});
            }
            report["rounds"].push_back(steps);
            trace.sample();
        }
        if (traceNumerics) {
            require(checkpoints.size() == (mode == "gelu" ? 5 : 6), "Missing expected checkpoint");
            report["checkpoints"] = checkpoints;
        }
        net->releaseSession(session);
        report["success"] = true;
    } catch (const std::exception& error) {
        report["error"] = error.what();
    }
    report["rssPeakMiB"] = trace.maxRss.load() / 1024.;
    report["minimumAvailableMiB"] = trace.minAvailable.load() / 1024.;
    std::ofstream(output + "/result.json") << report.dump(2);
    std::cout << report.dump() << '\n';
    return report["success"] == true ? 0 : 1;
}
