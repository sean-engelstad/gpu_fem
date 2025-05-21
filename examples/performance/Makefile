include ../../Makefile.in

default: 2_linear_static

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK} # --maxrregcount=32

%.ptx: %.cu
	${GPU_CXX} -ptx --source-in-ptx -g -G ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@ ${LINK} # --maxrregcount=32