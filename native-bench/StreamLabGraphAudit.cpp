// Read-only, pinned-schema audit for experimental batch/head optimization.
#include <MNN_generated.h>
#include <json.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
using Json=nlohmann::json;
int main(int argc,char** argv){
    if(argc!=2)return 2;std::ifstream file(argv[1],std::ios::binary|std::ios::ate);if(!file)return 2;
    std::vector<char> bytes(size_t(file.tellg()));file.seekg(0);file.read(bytes.data(),bytes.size());
    std::unique_ptr<MNN::NetT> net(MNN::GetNet(bytes.data())->UnPack());Json rows=Json::array();
    std::map<int,std::vector<std::string>> consumers;
    for(const auto& op:net->oplists)for(int id:op->inputIndexes)consumers[id].push_back(op->name+":"+MNN::EnumNameOpType(op->type));
    for(const auto& op:net->oplists){Json r={{"name",op->name},{"type",MNN::EnumNameOpType(op->type)}};
        bool keep=false;
        if(auto input=op->main.AsInput()){r["dims"]=input->dims;keep=true;}
        if(auto blob=op->main.AsBlob()){if(!blob->int32s.empty()&&blob->int32s.size()<=10){r["int32"]=blob->int32s;r["consumers"]=consumers[op->outputIndexes[0]];keep=true;}
            if(blob->dims.size()>=2&&blob->dims[0]==2){r["blobDims"]=blob->dims;r["intCount"]=blob->int32s.size();r["floatCount"]=blob->float32s.size();r["external"]=blob->external;r["consumers"]=consumers[op->outputIndexes[0]];keep=true;}}
        if(auto shape=op->main.AsReshape()){r["dims"]=shape->dims;keep=true;}
        if(auto conv=op->main.AsConvolution2D()){if(conv->common->outputCount>16000){r["inputCount"]=conv->common->inputCount;r["outputCount"]=conv->common->outputCount;r["external"]=conv->external;keep=true;
            if(auto q=conv->quanParameter.get())r["quant"]={{"type",q->type},{"readType",q->readType},{"aMin",q->aMin},{"bits",q->aMaxOrBits},{"shapeInt32",q->shapeInt32},{"alphaCount",q->alpha.size()},{"bufferCount",q->buffer.size()}};}}
        if(keep)rows.push_back(r);
    }
    std::cout<<rows.dump(2)<<std::endl;return 0;
}
