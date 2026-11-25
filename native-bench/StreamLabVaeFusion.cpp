// Real graph-level Snake fusion, no callbacks/snapshots in timed forwards.
#include "../runtime/include/levo/experimental/StreamLab.hpp"
#include "../runtime/include/levo/kernels/SnakeFusionGraph.hpp"
#ifdef LEVO_VAE_LAYOUT_ALIAS
#include "VaeLayoutAlias.hpp"
#endif

namespace {
class Decoder {
    std::unique_ptr<MNN::Interpreter> net_; MNN::Session* session_ = nullptr;
    void copyInput(const std::vector<float>& latent) {
        auto input = net_->getSessionInput(session_, nullptr);
        MNN::Tensor host(input, MNN::Tensor::CAFFE, true);
        require(host.elementSize() == latent.size(), "Latent size mismatch");
        std::memcpy(host.host<float>(), latent.data(), latent.size() * sizeof(float));
        require(input->copyFromHostTensor(&host), "Input copy failed");
    }
    std::vector<float> copyOutput() {
        auto output = net_->getSessionOutput(session_, nullptr);
        MNN::Tensor pcm(output, MNN::Tensor::CAFFE, true);
        require(output->copyToHostTensor(&pcm), "PCM read failed");
        return {pcm.host<float>(), pcm.host<float>() + pcm.elementSize()};
    }
public:
    Decoder(const std::string& path, const std::vector<char>* graph, Json& setup, bool profile) {
        auto tick = Clock::now(); net_.reset(graph ? MNN::Interpreter::createFromBuffer(graph->data(), graph->size()) : MNN::Interpreter::createFromFile(path.c_str()));
        require(bool(net_), "Cannot load decoder"); net_->setExternalFile((path + ".weight").c_str());
        if (profile) net_->setSessionMode(MNN::Interpreter::Session_Debug);
        net_->setSessionMode(MNN::Interpreter::Session_Resize_Defer); net_->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI, 0);
        MNN::BackendConfig backend; backend.precision = MNN::BackendConfig::Precision_Low; backend.memory = MNN::BackendConfig::Memory_Low; backend.power = MNN::BackendConfig::Power_High;
        MNN::ScheduleConfig config; config.type = MNN_FORWARD_CPU; config.numThread = 6; config.backendConfig = &backend;
        session_ = net_->createSession(config); require(session_, "Cannot create decoder");
        {LevoVaeSin::Scope sin(true); net_->resizeSession(session_);} net_->releaseModel();
        auto out = net_->getSessionOutput(session_, nullptr); require(out && out->elementSize() == 2 * 491520, "VAE256 shape failed");
        float memory = 0; net_->getSessionInfo(session_, MNN::Interpreter::MEMORY, &memory);
        setup = {{"seconds", seconds(tick)}, {"mnnSessionMemoryMiB", memory}, {"rssMiB", ::resident() / 1024.}};
    }
    ~Decoder() { if (session_) net_->releaseSession(session_); }
    std::vector<float> run(const std::vector<float>& latent, Json& row) {
        auto tick = Clock::now(); copyInput(latent);
        const double upload = seconds(tick); tick = Clock::now();
        const auto before = LevoSnakeFused::calls.load(); require(net_->runSession(session_) == 0, "VAE forward failed"); const auto compute = seconds(tick); tick = Clock::now();
        auto values = copyOutput(); const auto download = seconds(tick);
        row = {{"inputSeconds", upload}, {"forwardSeconds", compute}, {"outputSeconds", download}, {"totalSeconds", upload + compute + download},
               {"peak", peak(values)}, {"customCalls", LevoSnakeFused::calls.load() - before}, {"rssMiB", ::resident() / 1024.}}; return values;
    }
    Json profile(const std::vector<float>& latent, const std::vector<float>& reference) {
        Json tensors = Json::array(), commands = Json::array();
        std::map<const MNN::Tensor*, size_t> versions;
        auto tensorId = [&](const MNN::Tensor* tensor, bool produced) {
            auto known = versions.find(tensor);
            if (!produced && known != versions.end()) return known->second;
            auto describe = MNN::TensorUtils::getDescribe(tensor);
            auto backend = MNN::TensorUtils::getDescribeOrigin(tensor)->getBackend();
            size_t bytes = 0;
            if (backend && (backend->type() == MNN_FORWARD_CPU || backend->type() == MNN_FORWARD_CPU_EXTENSION)) {
                bytes = static_cast<const MNN::CPUBackend*>(backend)->getTensorSize(tensor, true);
            }
            const size_t id = tensors.size();
            versions[tensor] = id;
            tensors.push_back({{"shape", tensor->shape()}, {"format", MNN::EnumNameMNN_DATA_FORMAT(describe->dimensionFormat)},
                               {"storageBytes", bytes}, {"backend", backend ? int(backend->type()) : -1}});
            return id;
        };
        Json row;
        Clock::time_point start;
        bool failed = false;
        auto before = [&](const std::vector<MNN::Tensor*>& inputs, const MNN::OperatorInfo* info) {
            try {
                require(row.is_null(), "Overlapping operator callbacks");
                Json ids = Json::array();
                for (auto input : inputs) ids.push_back(tensorId(input, false));
                row = {{"name", info->name()}, {"type", info->type()}, {"inputIndexes", ids}};
                start = Clock::now();
            } catch (...) { failed = true; }
            return true;
        };
        auto after = [&](const std::vector<MNN::Tensor*>& outputs, const MNN::OperatorInfo* info) {
            const auto elapsed = seconds(start);
            try {
                require(row.at("name") == info->name(), "Mismatched operator callback");
                Json ids = Json::array(), regions = Json::array();
                if (info->type() == "Raster") {
                    for (auto output : outputs) {
                        for (const auto& region : MNN::TensorUtils::getDescribe(output)->regions) {
                            if (!region.origin) continue;
                            regions.push_back({{"source", tensorId(region.origin, false)}, {"size", region.size},
                                               {"sourceStride", region.src.stride}, {"outputStride", region.dst.stride},
                                               {"sourceOffset", region.src.offset}, {"outputOffset", region.dst.offset}});
                        }
                    }
                }
                for (auto output : outputs) ids.push_back(tensorId(output, true));
                row["outputIndexes"] = ids;
                row["rasterRegions"] = regions;
                row["seconds"] = elapsed;
                row["mnnMflops"] = info->flops();
                commands.push_back(std::move(row));
                row = nullptr;
            } catch (...) { failed = true; }
            return true;
        };
        copyInput(latent);
        const auto tick = Clock::now();
        require(net_->runSessionWithCallBackInfo(session_, before, after, true) == MNN::NO_ERROR,
                "VAE profile forward failed");
        const auto total = seconds(tick);
        require(!failed && row.is_null() && !commands.empty(), "VAE profile callback failure");
        require(copyOutput() == reference, "Profiled PCM differs from the unprofiled run");
        const size_t output = tensorId(net_->getSessionOutput(session_, nullptr), false);
        return {{"profiledForwardSeconds", total}, {"pcmMatchesWarmRun", true}, {"tensors", tensors},
                {"commands", commands}, {"graphOutputs", {output}}};
    }
};
}
int main(int argc, char** argv) {
    const bool once = argc == 6 && std::string(argv[5]) == "--once";
    bool profile = argc == 6 && std::string(argv[5]) == "--profile";
    bool sessionDebug = profile;
#ifdef LEVO_VAE_LAYOUT_ALIAS
    const bool aliasPlan = argc == 7 && std::string(argv[5]) == "--alias-plan";
    sessionDebug |= aliasPlan;
#endif
    if (argc != 5 && !sessionDebug && !once) return 2;
    const std::string model = std::string(argv[1]) + "/vae/vae.mnn", fixture = argv[2], output = argv[3], mode = argv[4];
    Json report = {{"success", false}, {"mode", mode}, {"scope", "Current real VAE graph-level Snake fusion; original W8 weights, FP16 CPU6, NEON sine; no App deployment"}}; Trace trace;
    try {
        require(mode == "baseline" || mode == "fused", "Unknown mode"); require(!std::filesystem::exists(output), "Preserve previous output"); std::filesystem::create_directories(output);
#ifdef LEVO_VAE_LAYOUT_ALIAS
        std::vector<std::string> aliases;
        int rounds = once ? 1 : 4;
        if (aliasPlan) {
            require(mode == "fused", "Alias plans use the deployed Snake baseline");
            std::ifstream input(argv[6]);
            Json plan; input >> plan;
            aliases = plan.at("rasterNames").get<std::vector<std::string>>();
            rounds = plan.value("rounds", 4);
            profile = plan.value("captureProfile", false);
            require(rounds >= 2 && rounds <= 6, "Invalid benchmark rounds");
            LevoVaeAlias::configure(aliases);
            report["requestedAliases"] = aliases;
        }
#else
        const int rounds = once ? 1 : 4;
#endif
        auto latent = read(fixture, 64 * 256); Json rewriteReport; auto graph = LevoSnakeFused::rewrite(model, rewriteReport); report["rewrite"] = rewriteReport;
        LevoSnakeFused::ensureRegistered();
        Decoder decoder(model, mode == "fused" ? &graph : nullptr, report["setup"], sessionDebug);
#ifdef LEVO_VAE_LAYOUT_ALIAS
        if (aliasPlan) {
            auto installed = LevoVaeAlias::installed();
            require(LevoVaeAlias::setupError().empty(), LevoVaeAlias::setupError());
            require(installed.size() == aliases.size(), "Missing alias commands at resize");
            report["installedAliases"] = installed;
        }
#endif
        report["runs"] = Json::array(); std::vector<float> reference;
        for (int repeat = 0; repeat < rounds; ++repeat) {
#ifdef LEVO_VAE_LAYOUT_ALIAS
            const auto before = LevoVaeAlias::calls();
#endif
            LevoStreamLab::check(); Json row; auto pcm = decoder.run(latent, row); row["mode"] = repeat ? "warm" : "first";
            if (mode == "fused") require(row["customCalls"] == 36, "Fused node fallback or missing chain");
#ifdef LEVO_VAE_LAYOUT_ALIAS
            if (aliasPlan) {
                row["aliasCalls"] = LevoVaeAlias::calls() - before;
                require(row["aliasCalls"] == aliases.size(), "Alias fallback or missing copy");
            }
#endif
            if (repeat == 0) {reference = pcm; std::ofstream file(output + "/pcm.f32", std::ios::binary); file.write(reinterpret_cast<const char*>(pcm.data()), pcm.size() * 4);}
            row["sameSessionRepeatL2"] = relative(pcm, reference); report["runs"].push_back(row); trace.sample();
            std::ofstream(output + "/result.json") << report.dump(2); std::cout << "VAE_FUSION_RUN " << row.dump() << std::endl;
        }
        if (profile) std::ofstream(output + "/profile.json") << decoder.profile(latent, reference).dump(2);
        report["success"] = true; report["fusedElements"] = LevoSnakeFused::elements.load();
    } catch (const std::exception& e) {report["error"] = e.what();}
#ifdef LEVO_VAE_LAYOUT_ALIAS
    if (!LevoVaeAlias::setupError().empty()) report["aliasSetupError"] = LevoVaeAlias::setupError();
#endif
    trace.sample(); report["rssPeakMiB"] = trace.maxRss.load() / 1024.; report["minimumAvailableMiB"] = trace.minAvailable.load() / 1024.;
    if (std::filesystem::is_directory(output)) std::ofstream(output + "/result.json") << report.dump(2);
    std::cout << "VAE_FUSION_DONE " << report.dump() << std::endl; return report["success"] == true ? 0 : 1;
}
