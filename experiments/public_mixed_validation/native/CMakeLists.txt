cmake_minimum_required(VERSION 3.15)
project(LevoPublicMixed CXX)
set(CMAKE_CXX_STANDARD 17)
set(MNN_ROOT "C:/cosyvoice-harmony/third_party/MNN")
set(MNN_LIBRARY "C:/cosyvoice-harmony/app/entry/.cxx/default/default/debug/arm64-v8a/mnn/libMNN.a")
add_executable(levo_public_codec PublicMixedCodec.cpp)
add_executable(levo_public_phone PublicMixedPhone.cpp)
foreach(public_target levo_public_codec levo_public_phone)
target_include_directories(${public_target} PRIVATE
  "C:/levo-harmony/native-bench" "${MNN_ROOT}/include" "${MNN_ROOT}/source"
  "${MNN_ROOT}/schema/current" "${MNN_ROOT}/3rd_party/flatbuffers/include"
  "${MNN_ROOT}/transformers/llm/engine/include" "${MNN_ROOT}/transformers/llm/engine/app/jsonhpp")
target_compile_definitions(${public_target} PRIVATE LEVO_VAE_VECTOR_SIN MNN_USE_THREAD_POOL MNN_SUPPORT_QUANT_EXTEND)
target_compile_options(${public_target} PRIVATE -O3 -march=armv8.2-a+fp16)
target_link_options(${public_target} PRIVATE -Wl,--gc-sections -Wl,-s)
target_link_libraries(${public_target} PRIVATE "${MNN_LIBRARY}" hilog_ndk.z)
endforeach()
