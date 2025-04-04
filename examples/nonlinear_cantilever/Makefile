include ../../Makefile.in

default: gpu

gpu:
	@echo "Building uCRM example for GPU"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 nl-static.cu ${LINK}