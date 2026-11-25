#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace LevoVaeAlias {
// Controls the one fixed-shape CPU VAE session created by the isolated CLI.
void configure(const std::vector<std::string>& names);
std::vector<std::string> installed();
std::string setupError();
uint64_t calls();
}
