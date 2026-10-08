// Compiler v0 native encoder/decoder. Must produce exactly the IDs that the Python
// reference (compiler/compiler_v0.py) defines; see that file for the full specification.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cv0 {

constexpr uint32_t CAP_ID = 256, UPPER_ID = 257, FIRST_ENTRY_ID = 258;
constexpr size_t MAX_ENTRY_LEN = 32, MAX_PHRASE_UNITS = 8;

inline bool letter(uint8_t c) { return (uint8_t)((c | 32) - 'a') < 26; }
inline bool upper(uint8_t c) { return (uint8_t)(c - 'A') < 26; }
inline bool lower(uint8_t c) { return (uint8_t)(c - 'a') < 26; }

inline uint64_t fnv1a64(const uint8_t* p, size_t n) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001B3ull;
    return h;
}

// Open-addressing map from byte strings to IDs; keys point into Dictionary storage.
class Table {
    struct Slot { const uint8_t* key = nullptr; uint32_t len = 0, id = 0; };
    std::vector<Slot> slots_;
    size_t mask_ = 0;
    static uint64_t hash(const uint8_t* p, size_t n) {
        uint64_t h = 0x9E3779B97F4A7C15ull ^ n;
        while (n >= 8) { uint64_t v; std::memcpy(&v, p, 8); h = (h ^ v) * 0xFF51AFD7ED558CCDull; h ^= h >> 32; p += 8; n -= 8; }
        uint64_t v = 0; std::memcpy(&v, p, n); h = (h ^ v) * 0xC4CEB9FE1A85EC53ull; h ^= h >> 29;
        return h;
    }
public:
    void build(const std::vector<std::pair<std::string_view, uint32_t>>& items) {
        size_t cap = 16;
        while (cap < items.size() * 2 + 1) cap <<= 1;
        slots_.assign(cap, {});
        mask_ = cap - 1;
        for (auto& [k, id] : items) {
            size_t i = hash((const uint8_t*)k.data(), k.size()) & mask_;
            while (slots_[i].key) i = (i + 1) & mask_;
            slots_[i] = {(const uint8_t*)k.data(), (uint32_t)k.size(), id};
        }
    }
    // Returns the ID or UINT32_MAX when absent.
    uint32_t find(const uint8_t* p, size_t n) const {
        if (slots_.empty()) return UINT32_MAX;
        size_t i = hash(p, n) & mask_;
        while (slots_[i].key) {
            const Slot& s = slots_[i];
            if (s.len == n && std::memcmp(s.key, p, n) == 0) return s.id;
            i = (i + 1) & mask_;
        }
        return UINT32_MAX;
    }
};

class Dictionary {
public:
    bool case_flags = true;
    size_t max_phrase_units = 1, max_piece_len = 1;
    uint64_t hash = 0;
    std::vector<std::string> id_bytes;  // ID -> bytes (flags map to "")
    Table start, cont, phrase;
    Table phrase_heads;                 // first two units of every phrase: a quick reject
    std::vector<std::string> head_store;
    bool has_phrases = false;

    static std::string unescape(std::string_view t) {
        std::string out;
        for (size_t i = 0; i < t.size();) {
            if (t[i] != '\\') { out.push_back(t[i++]); continue; }
            if (i + 1 < t.size() && t[i + 1] == '\\') { out.push_back('\\'); i += 2; continue; }
            if (i + 3 < t.size() && t[i + 1] == 'x') { out.push_back((char)std::stoi(std::string(t.substr(i + 2, 2)), nullptr, 16)); i += 4; continue; }
            throw std::runtime_error("bad escape in dictionary");
        }
        return out;
    }

    void parse(const std::string& raw) {
        hash = fnv1a64((const uint8_t*)raw.data(), raw.size());
        std::vector<std::string_view> lines;
        size_t a = 0;
        for (size_t i = 0; i < raw.size(); ++i)
            if (raw[i] == '\n') { lines.emplace_back(raw.data() + a, i - a); a = i + 1; }
        if (a != raw.size() || lines.size() < 5 || lines[0] != "compiler-v0 dictionary" || lines[1] != "version 1")
            throw std::runtime_error("not a compiler-v0 dictionary");
        auto value = [&](size_t i, std::string_view key) {
            if (lines[i].substr(0, key.size() + 1) != std::string(key) + " ") throw std::runtime_error("bad dictionary header");
            return std::stoul(std::string(lines[i].substr(key.size() + 1)));
        };
        case_flags = value(2, "case_flags") != 0;
        max_phrase_units = value(3, "max_phrase_units");
        size_t count = value(4, "entries");
        if (lines.size() != 5 + count || max_phrase_units < 1 || max_phrase_units > MAX_PHRASE_UNITS)
            throw std::runtime_error("bad dictionary entry count");
        id_bytes.clear();
        for (int b = 0; b < 256; ++b) id_bytes.emplace_back(1, (char)b);
        id_bytes.emplace_back();
        id_bytes.emplace_back();
        std::vector<char> kinds;
        for (size_t i = 0; i < count; ++i) {
            std::string_view line = lines[5 + i];
            if (line.size() < 2 || line[1] != '\t') throw std::runtime_error("bad dictionary entry");
            kinds.push_back(line[0]);
            id_bytes.push_back(unescape(line.substr(2)));
        }
        std::vector<std::pair<std::string_view, uint32_t>> s, c, p;
        for (size_t i = 0; i < count; ++i) {
            uint32_t id = FIRST_ENTRY_ID + (uint32_t)i;
            const std::string& v = id_bytes[id];
            if (kinds[i] == 'S') s.push_back({v, id});
            else if (kinds[i] == 'C') c.push_back({v, id});
            else if (kinds[i] == 'P') p.push_back({v, id});
            else throw std::runtime_error("unknown dictionary entry kind");
            if (kinds[i] != 'P') max_piece_len = std::max(max_piece_len, v.size());
        }
        start.build(s); cont.build(c); phrase.build(p);
        has_phrases = !p.empty();
        // Any phrase match covers at least two units, so its first two units must be
        // the head of some phrase. Looking that up first skips most phrase probes.
        head_store.clear();
        head_store.reserve(p.size());
        std::vector<std::pair<std::string_view, uint32_t>> heads;
        for (auto& [v, id] : p) {
            const uint8_t* q = (const uint8_t*)v.data();
            size_t e1 = unit_end_of(q, v.size(), 0);
            if (e1 >= v.size()) continue;
            size_t e2 = unit_end_of(q, v.size(), e1);
            head_store.emplace_back(v.substr(0, e2));
        }
        std::sort(head_store.begin(), head_store.end());
        head_store.erase(std::unique(head_store.begin(), head_store.end()), head_store.end());
        for (auto& h : head_store) heads.push_back({h, 0});
        phrase_heads.build(heads);
    }

    static size_t unit_end_of(const uint8_t* d, size_t n, size_t i);

    uint32_t vocab_size() const { return (uint32_t)id_bytes.size(); }
};

// End of the scanner unit that starts at i (see the scanner rules in compiler_v0.py).
inline size_t unit_end(const uint8_t* d, size_t n, size_t i) {
    uint8_t b = d[i];
    if (b == ' ' && i + 1 < n && letter(d[i + 1])) {
        size_t j = i + 2;
        while (j < n && letter(d[j])) ++j;
        return j;
    }
    if (letter(b)) {
        size_t j = i + 1;
        while (j < n && letter(d[j])) ++j;
        return j;
    }
    if (b == ' ' || b == '\n' || b == '\t') {
        size_t j = i + 1;
        while (j < n && d[j] == b) ++j;
        if (b == ' ' && j < n && letter(d[j])) --j;  // the last space belongs to the word
        return j;
    }
    return i + 1;
}

inline size_t Dictionary::unit_end_of(const uint8_t* d, size_t n, size_t i) { return unit_end(d, n, i); }

// Case flag for a unit: 0, CAP_ID or UPPER_ID.
inline uint32_t case_flag(const uint8_t* u, size_t n) {
    size_t s = (n >= 2 && u[0] == ' ') ? 1 : 0;
    if (s >= n || !letter(u[s])) return 0;
    size_t len = n - s;
    bool all_upper = true, rest_lower = true;
    for (size_t i = s; i < n; ++i) {
        all_upper &= upper(u[i]);
        if (i > s) rest_lower &= lower(u[i]);
    }
    if (len >= 2 && all_upper) return UPPER_ID;
    if (upper(u[s]) && rest_lower) return CAP_ID;
    return 0;
}

class Encoder {
    const Dictionary& d_;
    std::string low_;
public:
    explicit Encoder(const Dictionary& d) : d_(d) {}

    void pieces(const uint8_t* u, size_t n, std::vector<uint32_t>& out) const {
        const Table* t = &d_.start;
        for (size_t pos = 0; pos < n;) {
            size_t len = std::min(d_.max_piece_len, n - pos);
            uint32_t id = UINT32_MAX;
            for (; len >= 2; --len) {
                id = t->find(u + pos, len);
                if (id != UINT32_MAX) break;
            }
            if (id == UINT32_MAX) { id = u[pos]; len = 1; }
            out.push_back(id);
            pos += len;
            t = &d_.cont;
        }
    }

    void encode(const uint8_t* d, size_t n, std::vector<uint32_t>& out) {
        out.reserve(out.size() + n / 3 + 16);
        const size_t K = d_.has_phrases ? d_.max_phrase_units : 1;
        size_t ends[MAX_PHRASE_UNITS + 1];
        uint32_t flags[MAX_PHRASE_UNITS];
        size_t have = 0;  // lookahead units computed beyond position i
        size_t i = 0;
        while (i < n) {
            // Fill the lookahead: ends[m] is the end of unit m starting at i.
            if (have == 0) {
                ends[0] = unit_end(d, n, i);
                flags[0] = d_.case_flags ? case_flag(d + i, ends[0] - i) : 0;
                have = 1;
            }
            uint32_t flag = flags[0];
            size_t consumed = 1;
            if (flag) {
                out.push_back(flag);
                size_t len = ends[0] - i;
                low_.assign((const char*)d + i, len);
                for (char& c : low_) if (upper((uint8_t)c)) c = char(c + 32);
                pieces((const uint8_t*)low_.data(), len, out);
            } else {
                bool matched = false;
                if (K > 1 && have < 2 && ends[0] < n) {
                    ends[1] = unit_end(d, n, ends[0]);
                    flags[1] = d_.case_flags ? case_flag(d + ends[0], ends[1] - ends[0]) : 0;
                    have = 2;
                }
                if (K > 1 && have >= 2 && flags[1] == 0 && d_.phrase_heads.find(d + i, ends[1] - i) != UINT32_MAX) {
                    while (have < K && ends[have - 1] < n) {
                        size_t s = ends[have - 1];
                        ends[have] = unit_end(d, n, s);
                        flags[have] = d_.case_flags ? case_flag(d + s, ends[have] - s) : 0;
                        ++have;
                    }
                    for (size_t k = have; k >= 2 && !matched; --k) {
                        bool clean = true;
                        for (size_t m = 1; m < k; ++m) clean &= flags[m] == 0;
                        if (!clean) continue;
                        uint32_t id = d_.phrase.find(d + i, ends[k - 1] - i);
                        if (id != UINT32_MAX) { out.push_back(id); consumed = k; matched = true; }
                    }
                }
                if (!matched) pieces(d + i, ends[0] - i, out);
            }
            i = ends[consumed - 1];
            // Shift the lookahead window.
            for (size_t m = consumed; m < have; ++m) { ends[m - consumed] = ends[m]; flags[m - consumed] = flags[m]; }
            have -= consumed;
        }
    }

    void decode(const uint32_t* ids, size_t count, std::string& out) const {
        uint32_t mode = 0;
        bool seen = false;
        for (size_t k = 0; k < count; ++k) {
            uint32_t x = ids[k];
            if (x == CAP_ID || x == UPPER_ID) {
                if (!d_.case_flags) throw std::runtime_error("case flag in a stream encoded without case flags");
                mode = x; seen = false; continue;
            }
            if (x >= d_.id_bytes.size()) throw std::runtime_error("ID outside the dictionary");
            const std::string& piece = d_.id_bytes[x];
            if (!mode) { out += piece; continue; }
            for (char ch : piece) {
                uint8_t c = (uint8_t)ch;
                if (mode) {
                    if (letter(c)) {
                        if ((mode == UPPER_ID || !seen) && lower(c)) c = uint8_t(c - 32);
                        seen = true;
                    } else if (seen || c != ' ') {
                        mode = 0;
                    }
                }
                out.push_back((char)c);
            }
        }
    }
};

}  // namespace cv0
