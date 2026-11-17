// common/checksum.h: FNV-1a 64-bit (SPEC 4.4, 17.1). Header-only so every program in both lanes
// uses this one implementation and the checksums cannot differ.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>

constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
constexpr uint64_t FNV_PRIME        = 1099511628211ULL;
constexpr int      FNV_HEX_LEN      = 17;  // 16 hex digits plus the terminating NUL

inline uint64_t fnv1a(const void* data, size_t bytes) {
    const unsigned char* p = (const unsigned char*)data;
    uint64_t h = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= FNV_PRIME;
    }
    return h;
}

inline void fnv1a_hex(uint64_t h, char out[FNV_HEX_LEN]) {
    snprintf(out, FNV_HEX_LEN, "%016llx", (unsigned long long)h);
}
