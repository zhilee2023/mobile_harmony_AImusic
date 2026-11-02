cmake_minimum_required(VERSION 3.15)
project(LevoMusic)
set(CMAKE_CXX_STANDARD 17)
set(LEVO_MNN_ROOT "C:/cosyvoice-harmony/third_party/MNN" CACHE PATH "Pinned MNN source")
set(LEVO_MNN_LIBRARY "C:/cosyvoice-harmony/app/entry/.cxx/default/default/debug/arm64-v8a/mnn/libMNN.a" CACHE FILEPATH "Harmony MNN archive")
get_filename_component(LEVO_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../" ABSOLUTE)
add_library(entry SHARED napi_init.cpp)
target_include_directories(entry PRIVATE "${LEVO_PROJECT_ROOT}/native-bench" "${LEVO_PROJECT_ROOT}/experiments/public_mixed_validation/native"
  "${LEVO_MNN_ROOT}/include" "${LEVO_MNN_ROOT}/source" "${LEVO_MNN_ROOT}/schema/current" "${LEVO_MNN_ROOT}/3rd_party/flatbuffers/include"
  "${LEVO_MNN_ROOT}/transformers/llm/engine/include" "${LEVO_MNN_ROOT}/transformers/llm/engine/src" "${LEVO_MNN_ROOT}/transformers/llm/engine/app/jsonhpp")
target_compile_definitions(entry PRIVATE LEVO_VAE_VECTOR_SIN MNN_USE_THREAD_POOL MNN_SUPPORT_QUANT_EXTEND)
target_compile_options(entry PRIVATE -O3 -march=armv8.2-a+fp16)
target_link_libraries(entry PRIVATE "${LEVO_MNN_LIBRARY}" libace_napi.z.so hilog_ndk.z)
