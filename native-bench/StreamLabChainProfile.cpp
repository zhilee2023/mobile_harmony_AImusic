// Full real stream chain with opt-in operator callbacks, not a replacement App.
#include "../runtime/include/levo/experimental/StreamLab.hpp"

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "appRoot choice frames request.json newOutputDir baseline|operators\n";
        return 2;
    }
    const std::string output = argv[5], mode = argv[6];
    Json report = {{"success", false}, {"mode", mode}, {"scope", "Full CLI generation through the actual Stream Lab runtime; no playback or App foreground timing claim"}};
    try {
        require(mode == "baseline" || mode == "operators", "Unknown profiling mode");
        require(!std::filesystem::exists(output), "Preserve previous profile output");
        std::filesystem::create_directories(output);
        const auto request = readJson(argv[4]);
        require(!request.value("prefetch", false), "Single-owner profiling requires token prefetch disabled");
        require(!request.value("lyrics", std::string()).empty(), "Use real lyrics, not random-code transport mode");
        LevoStreamProfile::Capture capture;
        capture.operators = mode == "operators";
        LevoStreamProfile::active = &capture;
        std::atomic<bool> done{false};
        std::thread observer([&] {
            while (!done.load()) {
                auto state = LevoStreamLab::snapshot();
                std::cout << "CHAIN_PROFILE_PROGRESS " << Json({{"phase", state.value("phase", "idle")},
                    {"frames", state.value("conditionFrames", 0)}, {"chunk", state.value("chunk", 0)},
                    {"status", state.value("status", "idle")}}).dump() << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
        Json result;
        try {
            result = LevoStreamLab::run(argv[1], std::stoi(argv[2]), std::stoi(argv[3]), false,
                request.at("lyrics"), request.at("npy"), request);
        } catch (...) {
            done = true; observer.join(); LevoStreamProfile::active = nullptr; throw;
        }
        done = true; observer.join(); LevoStreamProfile::active = nullptr;
        report["generation"] = result;
        report["profile"] = capture.report();
        report["success"] = result.value("status", "error") == "completed" && capture.error.empty() && capture.pending.empty();
        report["timingInterpretation"] = capture.operators
            ? "Instrumented full chain; callback/debug overhead included, not normal production RTF"
            : "Fresh process, original non-debug MNN execution, lightweight stage clocks; OS file cache uncontrolled";
        const auto run = result.value("runId", std::string());
        require(!run.empty(), "Missing actual generation ID");
        const auto source = std::filesystem::path(argv[1]) / "stream-runs" / run;
        for (const auto& name : {"result.json", "codes.json"})
            std::filesystem::copy_file(source / name, std::filesystem::path(output) / name);
        for (const auto& chunk : result.at("chunks")) {
            const auto file = std::filesystem::path(chunk.at("path").get<std::string>());
            std::filesystem::copy_file(file, std::filesystem::path(output) / file.filename());
        }
    } catch (const std::exception& e) { report["error"] = e.what(); }
    if (std::filesystem::is_directory(output)) std::ofstream(output + "/profile.json") << report.dump(2);
    std::cout << "CHAIN_PROFILE_DONE " << Json({{"success", report["success"]}, {"mode", mode}}).dump() << std::endl;
    return report["success"] == true ? 0 : 1;
}
