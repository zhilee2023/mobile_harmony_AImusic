#include <napi/native_api.h>
#include <sys/resource.h>
#include <fstream>
#include <sstream>
#include <string>

static napi_value Memory(napi_env env, napi_callback_info) {
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
       <<"\",\"safeAllocationCeilingTested\":false}";
    napi_value result; auto value=out.str(); napi_create_string_utf8(env,value.c_str(),value.size(),&result); return result;
}
static napi_value Init(napi_env env,napi_value exports){
    napi_property_descriptor prop={"memoryProbe",nullptr,Memory,nullptr,nullptr,nullptr,napi_default,nullptr};
    napi_define_properties(env,exports,1,&prop);return exports;
}
static napi_module mod={1,0,nullptr,Init,"entry",nullptr,{0}};
extern "C" __attribute__((constructor)) void Register(){napi_module_register(&mod);}
