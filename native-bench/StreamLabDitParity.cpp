#include "../runtime/include/levo/experimental/StreamLab.hpp"
int main(int argc,char** argv){
    if(argc!=4)return 2;Json report={{"success",false},{"scope","Matched conditional velocity, not a music/RTF test"}};
    try{
        LevoStreamLab::Graph dual(argv[1],false,6),single(argv[2],false,6);
        std::mt19937 rng(2307);std::normal_distribution<float> normal;Json rows=Json::array();bool passed=true;
        for(int valid:{256,183,64}){
            std::vector<float> one(256*1200),mask(256*256);for(auto& v:one)v=normal(rng)*.1f;
            for(int i=0;i<valid;++i)std::fill_n(mask.data()+i*256,valid,1.f);
            auto two=one;two.insert(two.end(),one.begin(),one.end());auto masks=mask;masks.insert(masks.end(),mask.begin(),mask.end());
            auto tick=Clock::now();auto reference=dual.run({{"inputs",two},{"attention_mask",masks},{"timestep",{.5f,.5f}}},2*256*64);double dualTime=seconds(tick);
            tick=Clock::now();auto actual=single.run({{"inputs",one},{"attention_mask",mask},{"timestep",{.5f}}},256*64);double singleTime=seconds(tick);
            reference.resize(256*64);double error=relative(actual,reference),maxDiff=0;for(size_t i=0;i<actual.size();++i)maxDiff=std::max(maxDiff,std::abs(double(actual[i])-reference[i]));
            rows.push_back({{"validFrames",valid},{"l2",error},{"maxAbsoluteDiff",maxDiff},{"referencePeak",peak(reference)},{"actualPeak",peak(actual)},{"dualSeconds",dualTime},{"singleSeconds",singleTime}});
            passed=passed&&error<1e-5;
        }
        report["rows"]=rows;report["success"]=passed;if(!passed)report["error"]="Single-branch velocity differs from dual conditional reference at1e-5 budget";
    }catch(const std::exception& e){report["error"]=e.what();}
    std::ofstream(argv[3])<<report.dump(2);std::cout<<report.dump()<<std::endl;return report["success"]==true?0:1;
}
