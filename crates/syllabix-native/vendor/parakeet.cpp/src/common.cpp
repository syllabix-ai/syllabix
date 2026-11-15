#include "common.hpp"

#include <cerrno>
#include <cstring>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace pk {

bool write_file_atomic(const std::string& path, const std::string& bytes, std::string* err) {
    auto fail = [&](const std::string& why) {
        if (err) *err = why;
        return false;
    };
    if (path.empty()) return fail("path is empty");
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return fail("cannot write " + tmp + ": " + std::strerror(errno));
    bool ok = bytes.empty() || std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    int e = ok ? 0 : errno;
    if (std::fclose(f) != 0 && ok) {
        ok = false;
        e = errno;
    }
    if (!ok) {
        std::remove(tmp.c_str());
        return fail("cannot write " + tmp + ": " + (e ? std::strerror(e) : "write failed"));
    }
#ifdef _WIN32
    // rename() on the MSVC runtime fails when the target exists.
    if (!MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const unsigned long code = GetLastError();
        std::remove(tmp.c_str());
        return fail("cannot replace " + path + " (Windows error " + std::to_string(code) + ")");
    }
#else
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        const int re = errno;
        std::remove(tmp.c_str());
        return fail("cannot replace " + path + ": " + std::strerror(re));
    }
#endif
    return true;
}

}  // namespace pk
