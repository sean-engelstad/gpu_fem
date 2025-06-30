include ../../Makefile.in

default: 1_fw_interp

GPU_CC_FLAGS := -O3 -arch=sm_86 -lineinfo

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}

profile1: 1_fw_interp
	rm -rf report.ncu-rep || echo "no previous nsight compute report.ncu-rep files"
	sudo /usr/local/cuda-12.6/bin/ncu --set full --export  report.ncu-rep ./1_fw_interp.out
	ncu-ui report.ncu-rep