#include <napi/native_api.h>
#include <sys/resource.h>
#include <fstream>
#include <sstream>
#include <string>
#include <atomic>
#include <memory>
#include <hidebug/hidebug.h>
#ifdef LEVO_WITH_MNN
#include "VaeBench.hpp"
#include "ChainBench.hpp"
#endif

static std::atomic<bool> busy{false};
struct VaeTask { napi_async_work work{};napi_deferred deferred{};std::string directory,precision,result;bool chain=false; };
static std::string ReadString(napi_env env,napi_value value){
    size_t size=0;if(napi_get_value_string_utf8(env,value,nullptr,0,&size)!=napi_ok||size>4096)return {};
    std::string text(size+1,'\0');napi_get_value_string_utf8(env,value,text.data(),text.size(),&size);text.resize(size);return text;
}
static napi_value Vae(napi_env env,napi_callback_info info){
    napi_value args[2];size_t count=2;napi_get_cb_info(env,info,&count,args,nullptr,nullptr);
    if(count!=2){napi_throw_type_error(env,nullptr,"Expected fixture directory and precision");return nullptr;}
    auto task=std::make_unique<VaeTask>();task->directory=ReadString(env,args[0]);task->precision=ReadString(env,args[1]);
    task->chain=task->precision=="serial"||task->precision=="keep-llm"||task->precision=="all"||task->precision=="all-w4"||task->precision=="llm-short"||task->precision=="llm-long"||task->precision=="chunks-5"||task->precision=="chunks-5-tuned"||task->precision=="chunks-5-w8-vsin";
    if(task->directory.empty()||(!task->chain&&task->precision!="high"&&task->precision!="low")){napi_throw_type_error(env,nullptr,"Invalid diagnostic options");return nullptr;}
    if(busy.exchange(true)){napi_throw_error(env,nullptr,"A numerical test is already running");return nullptr;}
    napi_value promise,name;napi_create_promise(env,&task->deferred,&promise);napi_create_string_utf8(env,"LevoVaeStream",NAPI_AUTO_LENGTH,&name);
    auto execute=[](napi_env,void* data){auto* job=static_cast<VaeTask*>(data);
#ifdef LEVO_WITH_MNN
      job->result=(job->precision=="chunks-5"||job->precision=="chunks-5-tuned"||job->precision=="chunks-5-w8-vsin")?LevoRunFiveChunks(job->directory,job->precision=="chunks-5-tuned",job->precision=="chunks-5-w8-vsin"?"w8-vsin":"fp32"):(job->chain?LevoRunChainBench(job->directory,job->precision):LevoRunVaeBench(job->directory,job->precision));
#else
      job->result="{\"success\":false,\"error\":\"This build has no linked MNN runtime\"}";
#endif
    };
    auto complete=[](napi_env env,napi_status status,void* data){std::unique_ptr<VaeTask> job(static_cast<VaeTask*>(data));napi_value value;
      if(status!=napi_ok)job->result="{\"success\":false,\"error\":\"Async worker failed\"}";
      napi_create_string_utf8(env,job->result.c_str(),job->result.size(),&value);napi_resolve_deferred(env,job->deferred,value);napi_delete_async_work(env,job->work);busy=false;};
    if(napi_create_async_work(env,nullptr,name,execute,complete,task.get(),&task->work)!=napi_ok){busy=false;napi_throw_error(env,nullptr,"Cannot create numerical worker");return nullptr;}
    if(napi_queue_async_work(env,task->work)!=napi_ok){napi_delete_async_work(env,task->work);busy=false;napi_throw_error(env,nullptr,"Cannot queue numerical worker");return nullptr;}
    task.release();return promise;
}

static napi_value Memory(napi_env env, napi_callback_info) {
    HiDebug_MemoryLimit appLimit{};HiDebug_NativeMemInfo appMemory{};
    OH_HiDebug_GetAppMemoryLimit(&appLimit);OH_HiDebug_GetAppNativeMemInfo(&appMemory);
    std::ifstream mem("/proc/meminfo");
    std::string line; unsigned long long total=0, available=0;
    while(std::getline(mem,line)) {
        std::istringstream in(line); std::string key; unsigned long long n=0; in>>key>>n;
        if(key=="MemTotal:")total=n; if(key=="MemAvailable:")available=n;
    }
    rlimit rss{}, space{}; const bool rssOK=getrlimit(RLIMIT_RSS,&rss)==0;
    const bool spaceOK=getrlimit(RLIMIT_AS,&space)==0;
    auto limit=[](bool ok,rlim_t value){return !ok?std::string("unavailable"):(value==RLIM_INFINITY?std::string("unlimited"):std::to_string(value));};
    std::ostringstream out;
    out<<"{\"totalKiB\":"<<total<<",\"availableKiB\":"<<available
       <<",\"rssLimitBytes\":\""<<limit(rssOK,rss.rlim_cur)<<"\",\"addressLimitBytes\":\""<<limit(spaceOK,space.rlim_cur)
       <<"\",\"appRssLimitKiB\":"<<appLimit.rssLimit<<",\"appVssLimitKiB\":"<<appLimit.vssLimit
       <<",\"processPssKiB\":"<<appMemory.pss<<",\"processRssKiB\":"<<appMemory.rss
       <<",\"reportedLimitAvailable\":"<<((total>0&&appLimit.rssLimit>0&&appLimit.rssLimit<=total)?"true":"false")
       <<",\"limitInterpretation\":\"Nonzero values above physical RAM may be unlimited sentinels; not an allocation guarantee\""
       <<",\"safeAllocationCeilingTested\":false}";
    napi_value result; auto value=out.str(); napi_create_string_utf8(env,value.c_str(),value.size(),&result); return result;
}
static napi_value Init(napi_env env,napi_value exports){
    napi_property_descriptor props[]={
      {"memoryProbe",nullptr,Memory,nullptr,nullptr,nullptr,napi_default,nullptr},
      {"runVaeStream",nullptr,Vae,nullptr,nullptr,nullptr,napi_default,nullptr},
      {"runThreeModuleChain",nullptr,Vae,nullptr,nullptr,nullptr,napi_default,nullptr}};
    napi_define_properties(env,exports,3,props);return exports;
}
static napi_module mod={1,0,nullptr,Init,"entry",nullptr,{0}};
extern "C" __attribute__((constructor)) void Register(){napi_module_register(&mod);}
