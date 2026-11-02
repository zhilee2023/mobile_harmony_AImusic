#pragma once
#include "tokenizer/tokenizer.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

class MusicConditionTables {
    int descriptor=-1;size_t bytes=0,base=0;void* mapped=MAP_FAILED;Json header;
public:
    explicit MusicConditionTables(const std::string& path){
        descriptor=open(path.c_str(),O_RDONLY);require(descriptor>=0,"Missing trained condition tables");
        struct stat s{};require(fstat(descriptor,&s)==0,"Cannot stat condition tables");bytes=s.st_size;
        uint64_t length=0;require(pread(descriptor,&length,8,0)==8&&length<1024*1024,"Invalid safetensors header");
        std::string text(length,'\0');require(pread(descriptor,text.data(),length,8)==ssize_t(length),"Short table header");
        header=Json::parse(text);base=8+length;mapped=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,descriptor,0);
        require(mapped!=MAP_FAILED,"Cannot map condition tables");
    }
    ~MusicConditionTables(){if(mapped!=MAP_FAILED)munmap(mapped,bytes);if(descriptor>=0)close(descriptor);}
    std::vector<float> row(const std::string& name,int index){
        const auto& info=header.at("audiolm.condition_provider.conditioners."+name);
        require(info["dtype"]=="BF16","Expected BF16 trained table");auto shape=info["shape"].get<std::vector<size_t>>();
        require(shape.size()==2&&shape[1]==2048&&index>=0&&size_t(index)<shape[0],"Condition row outside trained table");
        const size_t offset=base+info["data_offsets"][0].get<size_t>()+size_t(index)*4096;
        require(offset+4096<=bytes,"Condition row outside file");std::vector<float> result(2048);
        const auto* values=reinterpret_cast<const uint16_t*>(static_cast<const char*>(mapped)+offset);
        for(int k=0;k<2048;++k){uint32_t bits=uint32_t(values[k])<<16;std::memcpy(&result[k],&bits,4);}return result;
    }
};

static const std::vector<std::string> musicSections={"[verse]","[chorus]","[bridge]","[intro-short]","[intro-medium]","[intro-long]","[outro-short]","[outro-medium]","[outro-long]","[inst-short]","[inst-medium]","[inst-long]","[silence]"};
static std::vector<int> musicTokenize(MNN::Transformer::Tokenizer& tokenizer,const std::string& text,bool lyric){
    if(!lyric)return tokenizer.encode("<|im_start|>"+text);
    std::vector<int> ids={151644};size_t start=0;
    while(start<text.size()){
        size_t next=std::string::npos;int section=-1;
        for(size_t k=0;k<musicSections.size();++k){size_t pos=text.find(musicSections[k],start);if(pos<next){next=pos;section=k;}}
        std::string part=text.substr(start,next==std::string::npos?next:next-start);
        if(!part.empty()){auto piece=tokenizer.encode(part);ids.insert(ids.end(),piece.begin(),piece.end());}
        if(next==std::string::npos)break;ids.push_back(151646+section);start=next+musicSections[section].size();
    }
    return ids;
}
static void writeBf16Row(std::ofstream& out,const std::vector<float>& row){
    for(float value:row){uint32_t bits;std::memcpy(&bits,&value,4);bits+=0x7fff+((bits>>16)&1);uint16_t bf=bits>>16;out.write(reinterpret_cast<const char*>(&bf),2);}
}
static Json prepareMusicPrefix(const std::string& root,const std::string& lyrics,const std::string& style){
    auto start=Clock::now();std::unique_ptr<MNN::Transformer::Tokenizer> tokenizer(MNN::Transformer::Tokenizer::createTokenizer(root+"/lm/tokenizer.mtok"));
    require(bool(tokenizer),"Cannot load tokenizer");auto ids=musicTokenize(*tokenizer,lyrics,true),styles=musicTokenize(*tokenizer,style,false);
    require(!lyrics.empty()&&ids.size()<=600,"歌词过长，请精简后再生成");require(!style.empty()&&styles.size()<=100,"风格描述过长，请精简后再生成");
    MusicConditionTables tables(root+"/conditions.safetensors");
    const auto outputPath=root+"/lm/generation-embeddings-bf16.bin";
    std::ofstream out(outputPath,std::ios::binary|std::ios::trunc);std::ifstream music(root+"/music-embeddings-bf16.bin",std::ios::binary);
    require(out.good()&&music.good(),"Missing trained music embeddings");out<<music.rdbuf();require(out.tellp()==67117056,"Music embedding file size mismatch");
    std::vector<float> bos(2048);music.clear();music.seekg(16385*4096);std::vector<uint16_t> raw(2048);music.read(reinterpret_cast<char*>(raw.data()),4096);
    for(int k=0;k<2048;++k){uint32_t bits=uint32_t(raw[k])<<16;std::memcpy(&bos[k],&bits,4);}
    auto text=[&](const std::vector<int>& tokens,int length,bool lyric){int covered=0;
        for(int i=0;i<length;++i){int id=i<int(tokens.size())?tokens[i]:151643;
            if(i<int(tokens.size())&&lyric&&id>=151646&&id<=151658)covered=id-151645;
            auto values=tables.row((lyric?"description":"type_info")+std::string(".output_proj.weight"),id);
            if(lyric){auto structure=tables.row("description.structure_emb.weight",i<int(tokens.size())?covered:0);for(int k=0;k<2048;++k)values[k]+=structure[k];}
            writeBf16Row(out,values);
        }
    };
    for(int branch=0;branch<2;++branch){
        text(branch?std::vector<int>{151644}:ids,600,true);
        writeBf16Row(out,tables.row("prompt_audio.EOT_emb",0));auto pad=tables.row("prompt_audio.emb.0.weight",16385);
        for(int i=0;i<251;++i)writeBf16Row(out,pad);text(branch?std::vector<int>{151644}:styles,100,false);writeBf16Row(out,bos);
    }
    std::ifstream fixture(root+"/mapping-prefix.bf16",std::ios::binary);require(fixture.good(),"Missing numerical reference prefix");out<<fixture.rdbuf();
    require(out.tellp()==75055104&&out.good(),"Runtime embedding size/write mismatch");
    return {{"lyricTokenIds",ids},{"styleTokenIds",styles},{"seconds",seconds(start)},
        {"allConditionEmbeddingsTrained",true},{"lyrics",lyrics},{"style",style}};
}
