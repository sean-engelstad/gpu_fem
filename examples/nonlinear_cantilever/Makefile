include ../../Makefile.in

default: gpu

gpu:
	@echo "Building nl-beam example for GPU"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 nl-static.cu ${LINK}

aeroelastic:
	@echo "Building nl-beam example for GPU"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 aeroelastic-nl.cu ${LINK}