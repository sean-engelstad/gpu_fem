include ../../Makefile.in

default: gpu

gpu:
	@echo "Building uCRM example for GPU"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 linear_static.cu ${LINK}

load_mesh:
	${CXX} ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 load_mesh.cpp ${CPU_LINK}