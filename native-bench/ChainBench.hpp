#pragma once
#include <string>
std::string LevoRunChainBench(const std::string& directory, const std::string& policy);
std::string LevoRunFiveChunks(const std::string& directory, bool tuned = false,
                             const std::string& vaeVariant = "fp32");
// Explicit structure/loader experiment; default App entry remains five chunks.
std::string LevoRunChunks(const std::string& directory, int chunkCount,
                         const std::string& ditSubdirectory, bool overlapLoading,
                         int llmThreads = 4, bool overlapCompute = false);
