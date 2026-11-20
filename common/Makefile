# common/Makefile: builds gen_data and compare_c with the SPEC 12 CPU flags.
CXX      = g++
CXXFLAGS = -std=c++17 -O3 -mavx2 -mf16c -fopenmp -ffp-contract=off

all: gen_data compare_c

gen_data: gen_data.cpp fp32_gemm.cpp fp32_gemm.h params.h checksum.h
	$(CXX) $(CXXFLAGS) -o $@ gen_data.cpp fp32_gemm.cpp

compare_c: compare_c.cpp checksum.h
	$(CXX) $(CXXFLAGS) -o $@ compare_c.cpp

clean:
	rm -f gen_data compare_c

.PHONY: all clean
