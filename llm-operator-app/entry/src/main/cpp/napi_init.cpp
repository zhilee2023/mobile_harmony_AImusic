#include <napi/native_api.h>
#include "levo/operator_lab/Baseline.hpp"
using namespace LlmOperatorLab;
static napi_value value(napi_env env,const Json& data){napi_value result;const auto text=data.dump();napi_create_string_utf8(env,text.c_str(),text.size(),&result);return result;}
static std::string text(napi_env env,napi_value v){size_t size=0;require(napi_get_value_string_utf8(env,v,nullptr,0,&size)==napi_ok&&size<4096,"Invalid request string");std::string out(size+1,0);napi_get_value_string_utf8(env,v,out.data(),out.size(),&size);out.resize(size);return out;}
struct Task{napi_async_work work{};napi_deferred promise{};std::string root;int model=0;Json options,result;};
static void execute(napi_env,void* p){auto& task=*static_cast<Task*>(p);try{task.result=run(task.root,task.model,task.options);}catch(const std::exception& e){patch({{"status","error"},{"error",e.what()}});task.result=poll();}}
static void complete(napi_env env,napi_status status,void* p){std::unique_ptr<Task> task(static_cast<Task*>(p));busy=false;if(status!=napi_ok)patch({{"status","error"},{"error","Background worker failed"}});patch({{"busy",false}});napi_resolve_deferred(env,task->promise,value(env,poll()));napi_delete_async_work(env,task->work);}
static napi_value Start(napi_env env,napi_callback_info info){bool owned=false;try{
    size_t count=3;napi_value args[3];napi_get_cb_info(env,info,&count,args,nullptr,nullptr);require(count==3,"Expected root/model/options");
    auto task=std::make_unique<Task>();task->root=text(env,args[0]);require(napi_get_value_int32(env,args[1],&task->model)==napi_ok,"Model must be numeric");caseName(task->model);task->options=Json::parse(text(env,args[2]));
    bool expected=false;require(busy.compare_exchange_strong(expected,true),"An experiment is already running");owned=true;cancelled=false;
    {std::lock_guard<std::mutex> lock(stateMutex);state={{"busy",true},{"status","starting"},{"phase","prepare"},{"model",task->model},{"runTag",task->options.value("runTag",std::string())}};}
    napi_value promise,name;require(napi_create_promise(env,&task->promise,&promise)==napi_ok,"Cannot create result promise");napi_create_string_utf8(env,"LlmOperatorLab",NAPI_AUTO_LENGTH,&name);
    require(napi_create_async_work(env,nullptr,name,execute,complete,task.get(),&task->work)==napi_ok,"Cannot create worker");
    if(napi_queue_async_work(env,task->work)!=napi_ok){napi_delete_async_work(env,task->work);throw std::runtime_error("Cannot queue worker");}
    task.release();return promise;
}catch(const std::exception& e){if(owned)busy=false;napi_throw_error(env,nullptr,e.what());return nullptr;}}
static napi_value Poll(napi_env env,napi_callback_info){return value(env,poll());}
static napi_value Cancel(napi_env env,napi_callback_info){cancelled=true;return value(env,{{"cancelRequested",true}});}
static napi_value Inspect(napi_env env,napi_callback_info info){try{size_t count=2;napi_value args[2];napi_get_cb_info(env,info,&count,args,nullptr,nullptr);require(count==2,"Expected root/model");int model=0;napi_get_value_int32(env,args[1],&model);return value(env,inspect(text(env,args[0]),model));}catch(const std::exception& e){return value(env,{{"ready",false},{"error",e.what()}});}}
static napi_value Init(napi_env env,napi_value exports){napi_property_descriptor props[]={{"inspect",nullptr,Inspect,nullptr,nullptr,nullptr,napi_default,nullptr},{"start",nullptr,Start,nullptr,nullptr,nullptr,napi_default,nullptr},{"poll",nullptr,Poll,nullptr,nullptr,nullptr,napi_default,nullptr},{"cancel",nullptr,Cancel,nullptr,nullptr,nullptr,napi_default,nullptr}};napi_define_properties(env,exports,4,props);return exports;}
static napi_module module={1,0,nullptr,Init,"entry",nullptr,{0}};
extern "C" __attribute__((constructor)) void RegisterLlmOperatorLab(){napi_module_register(&module);}
