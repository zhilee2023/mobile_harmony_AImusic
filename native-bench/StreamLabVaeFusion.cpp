// Real graph-level Snake fusion, no callbacks/snapshots in timed forwards.
#include "../runtime/include/levo/experimental/StreamLab.hpp"
#include "../runtime/include/levo/kernels/SnakeFusionGraph.hpp"

namespace {
class Decoder {
    std::unique_ptr<MNN::Interpreter> net_; MNN::Session* session_ = nullptr;
public:
    Decoder(const std::string& path, const std::vector<char>* graph, Json& setup) {
        auto tick = Clock::now(); net_.reset(graph ? MNN::Interpreter::createFromBuffer(graph->data(), graph->size()) : MNN::Interpreter::createFromFile(path.c_str()));
        require(bool(net_), "Cannot load decoder"); net_->setExternalFile((path + ".weight").c_str());
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
        auto tick = Clock::now(); auto input = net_->getSessionInput(session_, nullptr); MNN::Tensor host(input, MNN::Tensor::CAFFE, true);
        require(host.elementSize() == latent.size(), "Latent size mismatch"); std::memcpy(host.host<float>(), latent.data(), latent.size() * 4);
        require(input->copyFromHostTensor(&host), "Input copy failed"); const double upload = seconds(tick); tick = Clock::now();
        const auto before = LevoSnakeFused::calls.load(); require(net_->runSession(session_) == 0, "VAE forward failed"); const auto compute = seconds(tick); tick = Clock::now();
        auto output = net_->getSessionOutput(session_, nullptr); MNN::Tensor pcm(output, MNN::Tensor::CAFFE, true); require(output->copyToHostTensor(&pcm), "PCM read failed");
        std::vector<float> values(pcm.host<float>(), pcm.host<float>() + pcm.elementSize()); const auto download = seconds(tick);
        row = {{"inputSeconds", upload}, {"forwardSeconds", compute}, {"outputSeconds", download}, {"totalSeconds", upload + compute + download},
               {"peak", peak(values)}, {"customCalls", LevoSnakeFused::calls.load() - before}, {"rssMiB", ::resident() / 1024.}}; return values;
    }
};
}
int main(int argc, char** argv) {
    if (argc != 5) return 2; const std::string model = std::string(argv[1]) + "/vae/vae.mnn", fixture = argv[2], output = argv[3], mode = argv[4];
    Json report = {{"success", false}, {"mode", mode}, {"scope", "Current real VAE graph-level Snake fusion; original W8 weights, FP16 CPU6, NEON sine; no App deployment"}}; Trace trace;
    try {
        require(mode == "baseline" || mode == "fused", "Unknown mode"); require(!std::filesystem::exists(output), "Preserve previous output"); std::filesystem::create_directories(output);
        auto latent = read(fixture, 64 * 256); Json rewriteReport; auto graph = LevoSnakeFused::rewrite(model, rewriteReport); report["rewrite"] = rewriteReport;
        LevoSnakeFused::ensureRegistered();
        Decoder decoder(model, mode == "fused" ? &graph : nullptr, report["setup"]);
        report["runs"] = Json::array(); std::vector<float> reference;
        for (int repeat = 0; repeat < 4; ++repeat) {
            LevoStreamLab::check(); Json row; auto pcm = decoder.run(latent, row); row["mode"] = repeat ? "warm" : "first";
            if (mode == "fused") require(row["customCalls"] == 36, "Fused node fallback or missing chain");
            if (repeat == 0) {reference = pcm; std::ofstream file(output + "/pcm.f32", std::ios::binary); file.write(reinterpret_cast<const char*>(pcm.data()), pcm.size() * 4);}
            row["sameSessionRepeatL2"] = relative(pcm, reference); report["runs"].push_back(row); trace.sample();
            std::ofstream(output + "/result.json") << report.dump(2); std::cout << "VAE_FUSION_RUN " << row.dump() << std::endl;
        }
        report["success"] = true; report["fusedElements"] = LevoSnakeFused::elements.load();
    } catch (const std::exception& e) {report["error"] = e.what();}
    trace.sample(); report["rssPeakMiB"] = trace.maxRss.load() / 1024.; report["minimumAvailableMiB"] = trace.minAvailable.load() / 1024.;
    if (std::filesystem::is_directory(output)) std::ofstream(output + "/result.json") << report.dump(2);
    std::cout << "VAE_FUSION_DONE " << report.dump() << std::endl; return report["success"] == true ? 0 : 1;
}
