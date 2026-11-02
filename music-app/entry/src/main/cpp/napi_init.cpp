#include <napi/native_api.h>
#include "MusicRuntime.hpp"
#include "MusicUiBenchmark.hpp"
static std::atomic<bool> musicBusy{false};
static std::string jsText(napi_env env,napi_value value){size_t size=0;napi_get_value_string_utf8(env,value,nullptr,0,&size);require(size<32768,"Input too long");std::string text(size+1,'\0');napi_get_value_string_utf8(env,value,text.data(),text.size(),&size);text.resize(size);return text;}
static napi_value jsJson(napi_env env,const Json& value){napi_value result;auto text=value.dump();napi_create_string_utf8(env,text.c_str(),text.size(),&result);return result;}
struct MusicTask {napi_async_work work{};napi_deferred deferred{};std::string root,lyrics,style,name;int frames=750,fd=-1,ditSteps=2;bool importing=false,benchmark=false;Json result;~MusicTask(){if(fd>=0)close(fd);}};
static void executeMusic(napi_env,void* data){auto& task=*static_cast<MusicTask*>(data);try{
    if(task.benchmark){task.result=musicUiBenchmark(task.root);return;}
    if(!task.importing){task.result=musicGenerate(task.root,task.lyrics,task.style,task.frames,task.ditSteps);return;}
    const Json* asset=nullptr;for(const auto& item:musicAssetInventory())if(item["name"]==task.name)asset=&item;
    require(asset,"不是此模型所需的文件");std::filesystem::path dest=task.root+"/"+(*asset)["relative"].get<std::string>();std::filesystem::create_directories(dest.parent_path());
    std::string temp=dest.string()+".importing";std::ofstream out(temp,std::ios::binary);require(out.good(),"Cannot create imported model file");
    std::vector<char> buffer(1024*1024);uint64_t copied=0;ssize_t count=0;
    while((count=read(task.fd,buffer.data(),buffer.size()))>0){out.write(buffer.data(),count);copied+=count;require(out.good()&&copied<=(*asset)["bytes"].get<uint64_t>(),"模型文件大小不匹配");}
    require(count==0&&copied==(*asset)["bytes"].get<uint64_t>(),"模型文件不完整");out.close();std::filesystem::rename(temp,dest);task.result={{"success",true}};
}catch(const std::exception& e){task.result={{"success",false},{"error",e.what()}};}}
static void completeMusic(napi_env env,napi_status status,void* data){std::unique_ptr<MusicTask> task(static_cast<MusicTask*>(data));if(status!=napi_ok)task->result={{"success",false},{"error","后台任务未完成"}};napi_resolve_deferred(env,task->deferred,jsJson(env,task->result));napi_delete_async_work(env,task->work);musicBusy=false;}
static napi_value scheduleMusic(napi_env env,napi_callback_info info,bool importing,bool benchmark=false){
    size_t count=4;napi_value args[4];napi_get_cb_info(env,info,&count,args,nullptr,nullptr);
    try{require(count==(benchmark?1:(importing?3:4)),"Invalid request");auto task=std::make_unique<MusicTask>();task->root=jsText(env,args[0]);require(!task->root.empty(),"Missing model directory");
        task->importing=importing;task->benchmark=benchmark;if(importing){task->name=jsText(env,args[1]);int fd=-1;napi_get_value_int32(env,args[2],&fd);task->fd=dup(fd);require(task->fd>=0,"Cannot read selected file");}
        else if(!benchmark){task->lyrics=jsText(env,args[1]);task->style=jsText(env,args[2]);napi_get_value_int32(env,args[3],&task->frames);}
        require(!musicBusy.exchange(true),"已有任务正在运行");napi_value promise,label;napi_create_promise(env,&task->deferred,&promise);napi_create_string_utf8(env,"LevoMusic",NAPI_AUTO_LENGTH,&label);
        if(napi_create_async_work(env,nullptr,label,executeMusic,completeMusic,task.get(),&task->work)!=napi_ok){musicBusy=false;throw std::runtime_error("Cannot create background task");}
        if(napi_queue_async_work(env,task->work)!=napi_ok){napi_delete_async_work(env,task->work);musicBusy=false;throw std::runtime_error("Cannot start background task");}task.release();return promise;
    }catch(const std::exception& e){napi_throw_error(env,nullptr,e.what());return nullptr;}
}
static napi_value Generate(napi_env env,napi_callback_info info){return scheduleMusic(env,info,false);}
static napi_value UiBenchmark(napi_env env,napi_callback_info info){return scheduleMusic(env,info,false,true);}
static napi_value AckUiPhase(napi_env env,napi_callback_info info){size_t count=1;napi_value arg;napi_get_cb_info(env,info,&count,&arg,nullptr,nullptr);int phase=-1;if(count==1)napi_get_value_int32(env,arg,&phase);musicUiAcknowledged=phase;return jsJson(env,{{"success",true}});}
static napi_value Import(napi_env env,napi_callback_info info){return scheduleMusic(env,info,true);}
static napi_value Cancel(napi_env env,napi_callback_info){levoPublicCancelled=true;return jsJson(env,{{"success",true}});}
static napi_value Inspect(napi_env env,napi_callback_info info){size_t count=1;napi_value arg;napi_get_cb_info(env,info,&count,&arg,nullptr,nullptr);try{return jsJson(env,musicCheckAssets(jsText(env,arg)));}catch(const std::exception& e){return jsJson(env,{{"ready",false},{"error",e.what()}});}}
static napi_value Progress(napi_env env,napi_callback_info info){size_t count=1;napi_value arg;napi_get_cb_info(env,info,&count,&arg,nullptr,nullptr);Json result={{"busy",musicBusy.load()}};
    result["uiBenchmarkPhase"]=musicUiPhase.load();result["uiBenchmarkDecoded"]=musicUiDecoded.load();
    try{std::ifstream in(jsText(env,arg)+"/progress.jsonl");std::string line,last;while(std::getline(in,line))if(!line.empty())last=line;if(!last.empty())result["stage"]=Json::parse(last)["stage"];}catch(...){}return jsJson(env,result);}
static napi_value Init(napi_env env,napi_value exports){napi_property_descriptor props[]={
    {"generate",nullptr,Generate,nullptr,nullptr,nullptr,napi_default,nullptr},{"importAsset",nullptr,Import,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"inspect",nullptr,Inspect,nullptr,nullptr,nullptr,napi_default,nullptr},{"progress",nullptr,Progress,nullptr,nullptr,nullptr,napi_default,nullptr},{"cancel",nullptr,Cancel,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"uiBenchmark",nullptr,UiBenchmark,nullptr,nullptr,nullptr,napi_default,nullptr},{"ackUiPhase",nullptr,AckUiPhase,nullptr,nullptr,nullptr,napi_default,nullptr}};
    napi_define_properties(env,exports,7,props);return exports;}
static napi_module module={1,0,nullptr,Init,"entry",nullptr,{0}};
extern "C" __attribute__((constructor)) void RegisterMusic(){napi_module_register(&module);}
