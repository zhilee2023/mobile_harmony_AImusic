// Read-only graph export. The output may contain parameters: keep it private.
#include <MNN_generated.h>
#include <flatbuffers/minireflect.h>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0 || input.tellg() > 64 * 1024 * 1024) return 3;
    std::vector<uint8_t> data(static_cast<size_t>(input.tellg()));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(data.data()), data.size())) return 4;
    flatbuffers::Verifier verifier(data.data(), data.size());
    if (!MNN::VerifyNetBuffer(verifier)) return 5;
    std::ofstream output(argv[2], std::ios::binary);
    output << flatbuffers::FlatBufferToString(data.data(), MNN::NetTypeTable(), true);
    output.flush();
    if (!output) return 6;
    std::cout << "Graph exported; no session or inference created\n";
    return 0;
}
