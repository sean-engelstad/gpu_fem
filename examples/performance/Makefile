include ../../Makefile.in

GPU_CC_FLAGS := -O3 -arch=sm_86
# GPU_CC_FLAGS := -O3 -arch=sm_86 -Xptxas=-v -maxrregcount=128

default: linear_static
	./linear_static.out --just_assembly # for testing assembly right now


profile: linear_static
	rm -rf report.ncu-rep || echo "no previous nsight compute report.ncu-rep files"
	sudo /usr/local/cuda-12.6/bin/ncu --set full --export  report.ncu-rep ./linear_static.out --just_assembly
	ncu-ui report.ncu-rep

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}
