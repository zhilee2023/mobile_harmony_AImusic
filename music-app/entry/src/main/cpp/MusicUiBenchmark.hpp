#pragma once
// Debug-intent-only paired test. Never replaces the last song or music tokens.
inline std::atomic<int> musicUiPhase{-1},musicUiAcknowledged{-1},musicUiDecoded{0};

static Json musicUiTelemetry(){
    Json result={{"rssKiB",resident()},{"availableKiB",available()}};
    Json frequencies=Json::object(),thermal=Json::array();
    for(int i=0;i<16;++i){
        std::ifstream in("/sys/devices/system/cpu/cpufreq/policy"+std::to_string(i)+"/scaling_cur_freq");
        long value=0;if(in>>value)frequencies[std::to_string(i)]=value;
    }
    for(int i=0;i<80;++i){
        std::string base="/sys/class/thermal/thermal_zone"+std::to_string(i)+"/",name;
        std::ifstream type(base+"type"),temp(base+"temp");long value=0;
        if(std::getline(type,name)&&(temp>>value))thermal.push_back({{"zone",i},{"type",name},{"rawTemp",value}});
    }
    result["cpuFrequencyKHz"]=frequencies;result["thermalRaw"]=thermal;
    result["telemetryReadable"]=!frequencies.empty()||!thermal.empty();return result;
}

static Json musicUiBenchmark(const std::string& root){
    PublicPhoneTrace trace;trace.sink=root+"/progress.jsonl";std::ofstream(trace.sink).close();levoPublicCancelled=false;
    struct Restore{~Restore(){musicUiPhase=-1;musicUiAcknowledged=-1;musicUiDecoded=0;}} restore;
    Json report={{"success",false},{"appVersion","1.0.5"},{"kind","same-process-ui-ABA"},{"threads",4},{"precision","high"},
        {"bodyBits",4},{"headBits",8},{"prefixPositions",953},{"decodeCallsPerPhase",128},
        {"oneLoadedLlm",true},{"resetKvBeforeEachPhase",true},{"teacherForced",true},{"audioGenerated",false},
        {"scope","Warm full-prefix LLM only; A1 animation+800ms UI updates, B static UI, A2 animation+800ms UI updates. Phase polling and 20ms RSS sampling identical. No whole-chain RTF."},
        {"rows",Json::array()}};
    writePublicJson(root+"/last-ui-benchmark.json",report);
    try{
        trace.gate(2600);require(musicCheckAssets(root)["ready"],"Missing music assets");
        auto previous=readJson(root+"/phone-generated-codes.json");auto codes=previous["codes"].get<std::vector<int>>();
        require(codes.size()>=128,"Need at least128 saved real music codes");codes.resize(128);
        for(int token:codes)require(token>=0&&token<16384,"Invalid teacher code");report["teacherCodes"]=codes;
        auto config=readJson(root+"/lm/config-phone.json");
        config["thread_num"]=4;config["precision"]="high";config["reuse_kv"]=false;config["levo_lm_guidance"]=1;
        writePublicJson(root+"/lm/config-ui-benchmark.json",config);
        std::vector<int> prefix(953);std::iota(prefix.begin(),prefix.end(),16386);
        auto tick=Clock::now();PublicLlm model(Llm::createLLM(root+"/lm/config-ui-benchmark.json"),&Llm::destroy);
        require(model&&model->load(),"LLM load failed");report["loadSeconds"]=seconds(tick);
        trace.mark("ui-benchmark-full-prefix-warmup");tick=Clock::now();model->reset();model->generate_init();
        publicLogits(*model,prefix,true);for(int i=0;i<8;++i)publicLogits(*model,{codes[i]},false);
        report["warmupSeconds"]=seconds(tick);
        std::vector<std::vector<float>> references;references.reserve(129);
        for(int phase=1;phase<=3;++phase){
            trace.gate(1200);musicUiDecoded=0;musicUiAcknowledged=-1;musicUiPhase=phase;
            trace.mark("ui-benchmark-phase-"+std::to_string(phase));
            auto wait=Clock::now();while(musicUiAcknowledged.load()!=phase){
                require(!levoPublicCancelled.load(),"Benchmark cancelled");require(seconds(wait)<8,"UI phase acknowledgement timed out");
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(600));
            model->reset();model->generate_init();Json row={{"phase",phase},{"animationEnabled",phase!=2},{"telemetry",Json::array()}};
            // MNN applies the pending removal on the next forward. Its history
            // counter still reports the old cache until that forward completes.
            row["historyBeforePendingRemoval"]=model->getCurrentHistory();
            row["telemetry"].push_back(musicUiTelemetry());tick=Clock::now();auto logits=publicLogits(*model,prefix,true);
            row["prefillSeconds"]=seconds(tick);double maxL2=0;
            row["historyAfterPrefill"]=model->getCurrentHistory();require(model->getCurrentHistory()==953,"KV prefix length mismatch after reset");
            if(phase==1)references.push_back(logits);else maxL2=relative(logits,references[0]);
            row["telemetry"].push_back(musicUiTelemetry());std::vector<double> timings;
            for(int i=0;i<128;++i){
                trace.gate(900);tick=Clock::now();logits=publicLogits(*model,{codes[i]},false);timings.push_back(seconds(tick));
                if(phase==1)references.push_back(logits);else maxL2=std::max(maxL2,relative(logits,references[i+1]));
                musicUiDecoded=i+1;
                if((i+1)%32==0){trace.mark("ui-benchmark-phase-"+std::to_string(phase)+"-decode-"+std::to_string(i+1));row["telemetry"].push_back(musicUiTelemetry());}
            }
            double decode=std::accumulate(timings.begin(),timings.end(),0.);
            row["decodeSeconds"]=decode;row["decodeTokensPerSecond"]=128/decode;row["tokenSeconds"]=timings;
            row["historyAfter"]=model->getCurrentHistory();row["maxLogitsL2VsA1"]=maxL2;
            require(maxL2<1e-4,"UI benchmark numerical mismatch");report["rows"].push_back(row);
            writePublicJson(root+"/last-ui-benchmark.json",report);
        }
        report["success"]=true;model.reset();trace.mark("ui-benchmark-complete");
    }catch(const std::exception& e){report["error"]=e.what();}
    trace.sample();report["totalSeconds"]=seconds(trace.start);report["sampledPeakRssKiB"]=trace.maxRss.load();
    report["minimumSystemAvailableKiB"]=trace.minAvailable.load();report["stages"]=trace.stages;
    writePublicJson(root+"/last-ui-benchmark.json",report);return report;
}
