#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

int main() {
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t mib = 1024 * 1024;
    for (int workers : {1, 6}) {
        for (std::size_t bytes : {std::size_t(4 * 1024), std::size_t(64 * 1024), mib, 8 * mib, 24 * mib}) {
            const auto rounds = std::max<std::size_t>(4, 64 * mib / bytes);
            std::vector<double> rates;
            for (int sample = 0; sample < 6; ++sample) {
                std::vector<unsigned char> source(bytes * workers, 91), destination(bytes * workers);
                std::atomic<int> ready{0};
                std::atomic<bool> start{false};
                std::vector<Clock::time_point> ends(workers);
                std::vector<std::thread> threads;
                for (int worker = 0; worker < workers; ++worker) {
                    threads.emplace_back([&, worker] {
                        const auto* input = source.data() + worker * bytes;
                        auto* output = destination.data() + worker * bytes;
                        std::memcpy(output, input, bytes);
                        ++ready;
                        while (!start.load()) std::this_thread::yield();
                        for (std::size_t i = 0; i < rounds; ++i) {
                            std::memcpy(output, input, bytes);
                            asm volatile("" : : "r"(output) : "memory");
                        }
                        ends[worker] = Clock::now();
                    });
                }
                while (ready.load() != workers) std::this_thread::yield();
                const auto begin = Clock::now();
                start.store(true);
                for (auto& thread : threads) thread.join();
                if (source != destination) return 1;
                const auto elapsed = std::chrono::duration<double>(*std::max_element(ends.begin(), ends.end()) - begin);
                if (sample != 0) rates.push_back(2.0 * bytes * workers * rounds / elapsed.count() / 1e9);
            }
            std::sort(rates.begin(), rates.end());
            std::cout << workers << ',' << bytes << ',' << 2 * bytes * workers << ',' << rates[2] << '\n';
        }
    }
}
