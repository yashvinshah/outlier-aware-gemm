# gpu/Makefile: builds bench_gpu with the SPEC 12 GPU flags. sm_89 comes from the compute capability 8.9
# recorded in docs/env_gpu.md; the first build on ARC confirms that nvcc accepts it.
NVCC      = nvcc
NVCCFLAGS = -std=c++17 -O3 -arch=sm_89 --fmad=false --prec-div=true -Xcompiler "-O3 -ffp-contract=off"
SRCS      = bench_gpu.cu select.cu quant.cu int8_gemm.cu outlier_merge.cu fused.cu pertensor.cu fp32_cublas.cu util.cpp
HDRS      = stages.h util.h ../common/params.h ../common/checksum.h

bench_gpu: $(SRCS) $(HDRS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS) -lcublas

clean:
	rm -f bench_gpu

.PHONY: clean
