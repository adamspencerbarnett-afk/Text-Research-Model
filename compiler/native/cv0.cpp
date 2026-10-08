// Compiler v0 command-line tool: encode, decode and benchmark with a fitted dictionary.
//   cv0 encode DICT INPUT OUTPUT.ids
//   cv0 decode DICT INPUT.ids OUTPUT
//   cv0 bench  DICT INPUT [repeats]
// Build: g++ -O2 -std=c++17 -o cv0 cv0.cpp
#include "cv0.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

void write_file(const std::string& path, const std::string& data) {
    std::ofstream f(path, std::ios::binary);
    f.write(data.data(), (std::streamsize)data.size());
    if (!f) throw std::runtime_error("cannot write " + path);
}

template <class T> void put(std::string& s, T v) {
    for (size_t i = 0; i < sizeof(T); ++i) s.push_back(char((uint64_t)v >> (8 * i)));
}

template <class T> T get(const std::string& s, size_t at) {
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(T); ++i) v |= uint64_t((uint8_t)s[at + i]) << (8 * i);
    return (T)v;
}

// Header layout matches compiler_v0.ID_HEADER: "<4sBB2xQQ".
std::string pack_ids(const std::vector<uint32_t>& ids, const cv0::Dictionary& d) {
    uint8_t width = d.vocab_size() <= (1u << 16) ? 2 : 4;
    std::string out = "CV0I";
    out.push_back(1);
    out.push_back((char)width);
    out.push_back(0);
    out.push_back(0);
    put<uint64_t>(out, d.hash);
    put<uint64_t>(out, ids.size());
    out.reserve(out.size() + ids.size() * width);
    for (uint32_t x : ids) width == 2 ? put<uint16_t>(out, (uint16_t)x) : put<uint32_t>(out, x);
    return out;
}

std::vector<uint32_t> unpack_ids(const std::string& raw, const cv0::Dictionary& d) {
    if (raw.size() < 24 || raw.compare(0, 4, "CV0I") != 0 || raw[4] != 1) throw std::runtime_error("not a compiler-v0 ID file");
    uint8_t width = (uint8_t)raw[5];
    if (width != 2 && width != 4) throw std::runtime_error("bad ID width");
    if (get<uint64_t>(raw, 8) != d.hash) throw std::runtime_error("ID file was encoded with a different dictionary");
    uint64_t count = get<uint64_t>(raw, 16);
    if (raw.size() != 24 + count * width) throw std::runtime_error("ID file length does not match its header");
    std::vector<uint32_t> ids(count);
    for (uint64_t i = 0; i < count; ++i)
        ids[i] = width == 2 ? get<uint16_t>(raw, 24 + 2 * i) : get<uint32_t>(raw, 24 + 4 * i);
    return ids;
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            std::cerr << "usage: cv0 encode|decode DICT INPUT OUTPUT | cv0 bench DICT INPUT [repeats]\n";
            return 2;
        }
        std::string cmd = argv[1];
        cv0::Dictionary dict;
        dict.parse(read_file(argv[2]));
        cv0::Encoder enc(dict);
        std::string input = read_file(argv[3]);
        if (cmd == "encode" && argc == 5) {
            std::vector<uint32_t> ids;
            enc.encode((const uint8_t*)input.data(), input.size(), ids);
            write_file(argv[4], pack_ids(ids, dict));
            std::printf("{\"bytes\": %zu, \"ids\": %zu, \"vocab\": %u}\n", input.size(), ids.size(), dict.vocab_size());
            return 0;
        }
        if (cmd == "decode" && argc == 5) {
            std::vector<uint32_t> ids = unpack_ids(input, dict);
            std::string out;
            enc.decode(ids.data(), ids.size(), out);
            write_file(argv[4], out);
            return 0;
        }
        if (cmd == "bench") {
            int repeats = argc > 4 ? std::atoi(argv[4]) : 5;
            std::vector<uint32_t> ids;
            std::string back;
            double best_enc = 1e30, best_dec = 1e30;
            for (int r = 0; r < repeats; ++r) {
                ids.clear();
                auto t = std::chrono::steady_clock::now();
                enc.encode((const uint8_t*)input.data(), input.size(), ids);
                best_enc = std::min(best_enc, seconds_since(t));
                back.clear();
                back.reserve(input.size());
                t = std::chrono::steady_clock::now();
                enc.decode(ids.data(), ids.size(), back);
                best_dec = std::min(best_dec, seconds_since(t));
            }
            std::printf("{\"bytes\": %zu, \"ids\": %zu, \"ids_per_kb\": %.2f, \"encode_mb_s\": %.1f, "
                        "\"decode_mb_s\": %.1f, \"round_trip\": %s}\n",
                        input.size(), ids.size(), 1000.0 * ids.size() / std::max<size_t>(1, input.size()),
                        input.size() / best_enc / 1e6, input.size() / best_dec / 1e6,
                        back == input ? "true" : "false");
            return back == input ? 0 : 1;
        }
        std::cerr << "unknown command\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
