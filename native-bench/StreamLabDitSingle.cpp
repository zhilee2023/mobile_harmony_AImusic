// Rewrite only verified batch dimensions. All weight references stay unchanged.
// Use a separate output graph and require numerical parity before deployment.
#include <MNN_generated.h>
#include <json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
using Json=nlohmann::json;
int main(int argc,char** argv){
    if(argc!=3)return 2;Json report={{"success",false}};
    try{
        if(std::filesystem::exists(argv[2]))throw std::runtime_error("Preserve output graph");
        std::ifstream in(argv[1],std::ios::binary|std::ios::ate);if(!in)throw std::runtime_error("Missing graph");
        std::vector<char> bytes(size_t(in.tellg()));in.seekg(0);in.read(bytes.data(),bytes.size());
        std::unique_ptr<MNN::NetT> net(MNN::GetNet(bytes.data())->UnPack());
        std::map<int,std::vector<std::pair<MNN::OpT*,int>>> uses;
        for(auto& op:net->oplists)for(size_t i=0;i<op->inputIndexes.size();++i)uses[op->inputIndexes[i]].push_back({op.get(),int(i)});
        const std::vector<std::vector<int>> accepted={{2,6,-1},{2,256,3600},{2,256,30,-1,2},{2,256,30,-1},{2,256,30,40},{2,256,1200}};
        Json changes=Json::array();int inputs=0,shapes=0;
        for(auto& op:net->oplists){
            if(auto input=op->main.AsInput()){
                if((op->name!="inputs"&&op->name!="timestep"&&op->name!="attention_mask")||input->dims.empty()||input->dims[0]!=2)throw std::runtime_error("Unexpected DiT input");
                input->dims[0]=1;++inputs;changes.push_back(op->name);
            }
            if(auto blob=op->main.AsBlob())if(std::find(accepted.begin(),accepted.end(),blob->int32s)!=accepted.end()){
                for(const auto& use:uses[op->outputIndexes[0]])if(use.first->type!=MNN::OpType_Reshape||use.second!=1)throw std::runtime_error("Shared shape has a non-reshape consumer");
                blob->int32s[0]=1;++shapes;changes.push_back(op->name);
            }
        }
        if(inputs!=3||shapes!=6)throw std::runtime_error("Unexpected number of batch shape edits");
        flatbuffers::FlatBufferBuilder builder;auto value=MNN::Net::Pack(builder,net.get());builder.Finish(value);
        std::ofstream out(argv[2],std::ios::binary);out.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),builder.GetSize());out.close();
        if(!out.good())throw std::runtime_error("Write failed");report={{"success",true},{"batch",1},{"inputsChanged",inputs},{"shapeConstantsChanged",shapes},{"changed",changes},{"weightsChanged",false},{"numericParityRequired",true}};
    }catch(const std::exception& e){report["error"]=e.what();}
    std::cout<<report.dump()<<std::endl;return report["success"]==true?0:1;
}
