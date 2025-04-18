include ../../Makefile.in

FILES := 1_linear_static 2_nl_static 3_meld 4_coupled_demo

default: 1_linear_static

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}