# Shared ABI flags must match the pinned MNN archive used by both Apps.
function(levo_target_mnn target mnn_root mnn_library)
  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/include"
    "${mnn_root}/include" "${mnn_root}/source" "${mnn_root}/schema/current"
    "${mnn_root}/3rd_party/flatbuffers/include"
    "${mnn_root}/transformers/llm/engine/include"
    "${mnn_root}/transformers/llm/engine/src"
    "${mnn_root}/transformers/llm/engine/app/jsonhpp")
  target_compile_definitions(${target} PRIVATE LEVO_VAE_VECTOR_SIN MNN_USE_THREAD_POOL MNN_SUPPORT_QUANT_EXTEND)
  target_compile_options(${target} PRIVATE -O3 -march=armv8.2-a+fp16)
  target_link_libraries(${target} PRIVATE ${mnn_library} hilog_ndk.z)
endfunction()
