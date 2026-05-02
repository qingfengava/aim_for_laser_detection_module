#pragma once

#include <chrono>
#include <cstdlib>
#include <string>

namespace laser_aim::utils {

inline std::string expandEnv(const std::string& input) {
    std::string out;
    out.reserve(input.size());

    for (std::size_t i = 0; i < input.size(); ++i) {
        if (input[i] == '$' && i + 1 < input.size() && input[i + 1] == '{') {
            const auto end = input.find('}', i + 2);
            if (end == std::string::npos) {
                out.push_back(input[i]);
                continue;
            }
            const std::string key = input.substr(i + 2, end - (i + 2));
            const char* val = std::getenv(key.c_str());
            if (val != nullptr) {
                out += val;
            }
            i = end;
            continue;
        }
        out.push_back(input[i]);
    }
    return out;
}

template<class Tag, class Func>
inline void xSecOnce(Func&& func, double dt_sec) noexcept {
    static auto last_call = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - last_call).count();
    if (elapsed >= dt_sec) {
        last_call = now;
        func();
    }
}

} // namespace laser_aim::utils
