// common/compare_c.cpp: bitwise comparison of two raw float32 files (SPEC 13 step 5).
// Usage: compare_c a.f32 b.f32 <count>   (count = number of float32 elements in each file)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "checksum.h"

static void die(const char* what, const char* detail) {
    fprintf(stderr, "compare_c: %s: %s\n", what, detail);
    exit(1);
}

// A file whose size is not count * 4 bytes is an error, so the checksum always covers the whole file.
static float* read_floats(const char* path, size_t count) {
    size_t bytes = count * sizeof(float);
    FILE* f = fopen(path, "rb");
    if (f == nullptr) die("cannot open", path);
    if (fseek(f, 0, SEEK_END) != 0) die("seek failed", path);
    long size = ftell(f);
    if (size < 0) die("cannot get file size", path);
    if ((size_t)size != bytes) die("file size is not count * 4 bytes", path);
    if (fseek(f, 0, SEEK_SET) != 0) die("seek failed", path);
    float* buf = (float*)malloc(bytes > 0 ? bytes : 1);
    if (buf == nullptr) die("out of memory", path);
    if (fread(buf, 1, bytes, f) != bytes) die("read failed", path);
    if (fclose(f) != 0) die("close failed", path);
    return buf;
}

int main(int argc, char** argv) {
    if (argc != 4) die("usage", "compare_c a.f32 b.f32 <count>");
    char* end = nullptr;
    unsigned long long count = strtoull(argv[3], &end, 10);
    if (end == argv[3] || *end != '\0') die("count must be an integer", argv[3]);

    float* a = read_floats(argv[1], (size_t)count);
    float* b = read_floats(argv[2], (size_t)count);

    size_t differing = 0;
    double max_abs = 0.0;
    for (size_t i = 0; i < (size_t)count; i++) {
        uint32_t bits_a, bits_b;
        memcpy(&bits_a, &a[i], sizeof(bits_a));
        memcpy(&bits_b, &b[i], sizeof(bits_b));
        if (bits_a != bits_b) differing++;
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > max_abs) max_abs = d;
    }

    char hex_a[FNV_HEX_LEN], hex_b[FNV_HEX_LEN];
    fnv1a_hex(fnv1a(a, count * sizeof(float)), hex_a);
    fnv1a_hex(fnv1a(b, count * sizeof(float)), hex_b);
    printf("DIFFERING_ELEMENTS %zu\n", differing);
    printf("MAX_ABS_DIFF %.9e\n", max_abs);
    printf("CHECKSUM %s %s\n", argv[1], hex_a);
    printf("CHECKSUM %s %s\n", argv[2], hex_b);

    free(a);
    free(b);
    return 0;
}
