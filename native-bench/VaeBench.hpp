#pragma once
#include <string>
std::string LevoRunVaeBench(const std::string& directory, const std::string& precision,
                          int threads = 4, bool performanceCores = false,
                          const std::string& modelVariant = "fp32", bool normalMemory = false);
