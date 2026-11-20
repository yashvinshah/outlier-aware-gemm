// common/params.h: constants fixed by SPEC.md. Do not change without a SPEC change.
#pragma once
#include <cstdint>
constexpr float    OUTLIER_SCALE = 50.0f;   // true outlier channels of X are multiplied by this
constexpr double   W_SIGMA       = 0.02;    // standard deviation of W entries
constexpr uint64_t SEED_X        = 1234;
constexpr uint64_t SEED_W        = 5678;
constexpr uint64_t SEED_OUTLIERS = 91011;
constexpr float    INT8_MAX_F    = 127.0f;  // INT8 scale divisor
constexpr int      DEFAULT_TILE  = 16;      // GPU default
