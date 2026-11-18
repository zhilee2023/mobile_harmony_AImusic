// Experimental exact row crop of this pinned W8 music head only.
// Original graph/weights are immutable. Validate retained logits before use.
#include <MNN_generated.h>
#include <json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
using Json=nlohmann::json;
static void need(bool yes,const char* error){if(!yes)throw std::runtime_error(error);}
static void copy(std::ifstream& in,std::ofstream& out,uint64_t start,uint64_t bytes){
    in.seekg(start);char buffer[65536];while(bytes){auto n=std::min<uint64_t>(bytes,sizeof(buffer));in.read(buffer,n);need(in.gcount()==n,"Short source read");out.write(buffer,n);bytes-=n;}
}
int main(int argc,char** argv){
    if(argc!=3&&argc!=4)return 2;Json report={{"success",false}};
    try{
        const std::string source=argv[1],folder=argv[2];need(!std::filesystem::exists(folder),"Preserve output directory");std::filesystem::create_directories(folder);
        const bool audioOnly=argc==4;need(!audioOnly||std::string(argv[3])=="audio-only","Unknown head crop mode");
        const uint32_t first=151765,count=audioOnly?16385:16449;
        std::ifstream graph(source,std::ios::binary|std::ios::ate);need(bool(graph),"Missing graph");std::vector<char> bytes(size_t(graph.tellg()));graph.seekg(0);graph.read(bytes.data(),bytes.size());
        std::unique_ptr<MNN::NetT> net(MNN::GetNet(bytes.data())->UnPack());MNN::Convolution2DT* head=nullptr;int reshape=0;
        for(auto& op:net->oplists){if(auto c=op->main.AsConvolution2D())if(c->common->outputCount==168214){need(!head,"Multiple heads");head=c;}
            if(auto r=op->main.AsReshape())if(op->name=="/lm/lm_head/Linear/post_reshape"){need(r->dims.back()==168214,"Unexpected output reshape");r->dims.back()=count;++reshape;}}
        need(head&&reshape==1,"Missing expected head");auto q=head->quanParameter.get();
        need(head->common->inputCount==1024&&head->common->kernelX==1&&head->common->kernelY==1&&head->external.size()==5,"Unexpected head layout");
        need(q&&q->type==1&&q->aMaxOrBits==8&&q->shapeInt32&&q->buffer.empty()&&q->alpha.empty(),"Unsupported quantized storage");
        const auto ext=head->external;const uint64_t offset=ext[0],weightBytes=ext[1],alphaBytes=ext[2];
        need(ext[3]==0&&ext[4]==0&&std::filesystem::file_size(source+".weight")==offset+weightBytes+alphaBytes,"Head must be last, without external bias");
        need(weightBytes==uint64_t(168214)*1024+266&&alphaBytes%168214==0,"Unexpected W8 row encoding");
        std::ifstream input(source+".weight",std::ios::binary);input.seekg(offset);unsigned char header[266];input.read(reinterpret_cast<char*>(header),266);
        uint32_t rows=0,cols=0;std::memcpy(&rows,header+1,4);std::memcpy(&cols,header+5,4);
        need(header[0]==2&&rows==168214&&cols==1024,"Unexpected shape header");
        std::memcpy(header+1,&count,4);
        std::ofstream out(folder+"/llm.mnn.weight",std::ios::binary);copy(input,out,0,offset);out.write(reinterpret_cast<char*>(header),266);
        const std::vector<std::pair<uint32_t,uint32_t>> ranges=audioOnly?std::vector<std::pair<uint32_t,uint32_t>>{{first,1},{151830,16384}}:std::vector<std::pair<uint32_t,uint32_t>>{{first,count}};
        for(const auto& r:ranges)copy(input,out,offset+266+uint64_t(r.first)*1024,uint64_t(r.second)*1024);
        const uint64_t alphaRow=alphaBytes/168214;for(const auto& r:ranges)copy(input,out,offset+weightBytes+uint64_t(r.first)*alphaRow,uint64_t(r.second)*alphaRow);
        out.close();need(out.good(),"Output failed");
        head->common->outputCount=count;head->external[1]=uint64_t(count)*1024+266;head->external[2]=uint64_t(count)*alphaRow;
        flatbuffers::FlatBufferBuilder builder;auto value=MNN::Net::Pack(builder,net.get());builder.Finish(value);
        std::ofstream output(folder+"/llm.mnn",std::ios::binary);output.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),builder.GetSize());output.close();need(output.good(),"Graph write failed");
        report={{"success",true},{"firstRetainedClass",first},{"classes",count},{"audioOutputBase",audioOnly?1:65},{"eosOutput",0},{"sourceEos",151765},{"inputAudioBase",151830},{"retainedSourceRowRanges",ranges},{"audioOnly",audioOnly},
            {"sourceWeightBytes",offset+weightBytes+alphaBytes},{"weightBytes",std::filesystem::file_size(folder+"/llm.mnn.weight")},{"scope","Exact encoded row/alpha crop; numerical parity still required"}};
        std::ofstream(folder+"/head-range.json")<<report.dump(2);
    }catch(const std::exception& e){report["error"]=e.what();}
    std::cout<<report.dump()<<std::endl;return report["success"]==true?0:1;
}
