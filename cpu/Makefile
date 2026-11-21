# cpu/Makefile: builds bench_cpu with the SPEC 12 CPU flags.
CXX         = g++
CXXFLAGS    = -std=c++17 -O3 -mavx2 -mf16c -fopenmp -ffp-contract=off -I../common
SCALARFLAGS = -std=c++17 -O2 -fno-tree-vectorize -fno-tree-slp-vectorize -fopenmp -ffp-contract=off -I../common
OBJS        = bench_cpu.o util.o select.o quant.o pertensor.o int8_scalar.o int8_avx2.o int8_avx2_opt1.o fused_avx2.o fp32_gemm.o

bench_cpu: $(OBJS)
	$(CXX) -fopenmp -o $@ $(OBJS)

int8_scalar.o: int8_scalar.cpp kernels.h
	$(CXX) $(SCALARFLAGS) -c -o $@ int8_scalar.cpp

fp32_gemm.o: ../common/fp32_gemm.cpp ../common/fp32_gemm.h
	$(CXX) $(CXXFLAGS) -c -o $@ ../common/fp32_gemm.cpp

%.o: %.cpp kernels.h util.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f bench_cpu $(OBJS)

.PHONY: clean
