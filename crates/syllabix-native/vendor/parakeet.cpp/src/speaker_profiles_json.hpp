#pragma once
#include "speaker_identifier.hpp"
#include <iomanip>
#include <locale>
#include <sstream>

namespace pk {
// Shared production assembler: nullptr preserves the legacy JSON byte for byte.
// identity is generated internally; reasons are the fixed SpeakerProfile codes.
inline std::string named_profiles_json(std::string json, const std::string& names,
    const std::map<int, SpeakerProfile>* profiles, const std::string& identity, int dimension) {
    json.pop_back();
    json += ",\"names\":" + names;
    if (!profiles) return json + "}";
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(9);
    out << json << ",\"speaker_profiles\":{\"version\":1,\"encoder\":{\"identity\":\""
        << identity << "\",\"dimension\":" << dimension << "},\"speakers\":[";
    bool first = true;
    for (const auto& kv : *profiles) {
        if (!first) out << ',';
        first = false;
        const auto& p = kv.second;
        out << "{\"speaker\":" << kv.first << ",\"clean_duration\":" << p.clean_duration
            << ",\"intervals\":[";
        for (size_t i = 0; i < p.intervals.size(); ++i) {
            if (i) out << ',';
            out << "{\"start\":" << p.intervals[i].start << ",\"end\":" << p.intervals[i].end << '}';
        }
        out << "],\"unavailable_reason\":";
        if (p.unavailable_reason.empty()) out << "null";
        else out << '"' << p.unavailable_reason << '"';
        if (!p.embedding.empty()) {
            out << ",\"embedding\":[";
            for (size_t i = 0; i < p.embedding.size(); ++i) {
                if (i) out << ',';
                out << p.embedding[i];
            }
            out << ']';
        }
        out << '}';
    }
    return out.str() + "]}}";
}
} // namespace pk
