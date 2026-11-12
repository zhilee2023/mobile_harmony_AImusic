// Compare the new direct-embedding entry with the original App's saved ID entry.
// Uses existing read-only model/embedding files; no DiT, VAE or quality claim.
#include "../runtime/include/levo/experimental/StreamLab.hpp"
#include <numeric>

int main(int argc,char** argv) {
    if(argc!=5&&argc!=6){std::cerr<<"smallBundle originalGenerationEmbedding request.json newWorkDir [croppedLmDir]\n";return 2;}
    Json report={{"success",false},{"scope","Small prior input-entry parity, no audio generation"}};
    try {
        const std::string root=argv[1],embedding=argv[2],work=argv[4];const bool crop=argc==6;
        require(!std::filesystem::exists(work),"Preserve existing probe output");
        std::filesystem::create_directories(work);
        auto request=readJson(argv[3]);
        auto prompt=LevoStreamLab::prefix(root,1024,true,request.at("lyrics"),request.at("npy"));
        std::vector<int> prefixIds(513);std::iota(prefixIds.begin(),prefixIds.end(),168215);
        // Verify the exact rows before comparing the two engine entry points.
        LevoStreamLab::MappedFloats original(embedding);std::vector<float> saved(prompt.size());
        for(int i=0;i<513;++i)original.row(prefixIds[i],1024,saved.data()+size_t(i)*1024,true);
        report["prefixL2"]=relative(prompt,saved);require(prompt==saved,"Original saved prefix differs");
        std::vector<float> reference;std::vector<int> codes;
        for(int round=0;round<2;++round){
            Json cfg={{"base_dir",round==1&&crop?std::string(argv[5])+"/":root+"/lm/"},{"llm_model","llm.mnn"},{"llm_weight","llm.mnn.weight"},
                {"llm_config","llm_config.json"},{"tokenizer_file","tokenizer.mtok"},
                {"embedding_file",round==0?"generation-embeddings-bf16.bin":"embeddings_bf16.bin"},
                {"backend_type","cpu"},{"precision","high"},{"memory","low"},{"power","high"},
                {"thread_num",4},{"reuse_kv",false},{"attention_mode",8},{"hidden_states",false},
                {"all_logits",false},{"cosyvoice_kleidiai",false},{"tmp_path",work+"/"}};
            const auto path=work+"/config-"+std::to_string(round)+".json";
            {std::ofstream out(path);out<<cfg.dump(2);}
            std::unique_ptr<Llm,decltype(&Llm::destroy)> model(Llm::createLLM(path),&Llm::destroy);
            require(model&&model->load(),"Cannot load small prior");model->reset();model->generate_init();
            std::vector<float> output;std::vector<int> greedy;Json stats=Json::array();
            for(int step=0;step<9;++step){
                auto v=step?model->forward({151830+codes[step-1]},false):
                    (round==0?model->forward(prefixIds,true):model->forward(MNN::Express::_Const(prompt.data(),{513,1,1024},MNN::Express::NCHW,halide_type_of<float>())));
                const int classes=round==1&&crop?16449:168214,audioBase=round==1&&crop?65:151830;
                require(v.get()&&v->getInfo()->size==classes,"Unexpected small prior output");
                const float* p=v->readMap<float>();require(p,"Cannot read logits");
                double energy=0,maxAbs=0;size_t nonzero=0;for(int i=0;i<classes;++i){energy+=double(p[i])*p[i];maxAbs=std::max(maxAbs,std::abs(double(p[i])));nonzero+=p[i]!=0;}
                stats.push_back({{"step",step},{"peak",maxAbs},{"energy",energy},{"nonzero",nonzero},{"typeBits",v->getInfo()->type.bits}});
                output.insert(output.end(),p+(crop&&round==0?151765:0),p+classes);
                greedy.push_back(int(std::max_element(p+audioBase,p+classes)-(p+audioBase)));
                if(round==0)codes.push_back(greedy.back());
            }
            peak(output);report[round==0?"idEntryStats":"directEntryStats"]=stats;report[round==0?"idEntryGreedy":"directEntryGreedy"]=greedy;
            report[round==0?"idEntryKvPositions":"directEntryKvPositions"]=model->getCurrentHistory();
            if(round==0)reference=std::move(output);
            else {report["allLogitsL2"]=relative(output,reference);size_t different=0;
                for(size_t i=0;i<output.size();++i)different+=output[i]!=reference[i];report["differentLogitElements"]=different;}
        }
        report["success"]=true;
    }catch(const std::exception& error){report["error"]=error.what();}
    if(argc>=5){std::ofstream out(std::string(argv[4])+"/result.json");out<<report.dump(2);}
    std::cout<<"STREAM_PREFILL_REPORT "<<report.dump()<<std::endl;return report["success"]==true?0:1;
}
