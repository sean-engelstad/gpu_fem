include ../../Makefile.in

default: gpu

gpu:
	@echo "Building MELD example for GPU"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 test_transfer.cu ${LINK}

run:
	make gpu
	./a.out
	paraview xa0.vtk