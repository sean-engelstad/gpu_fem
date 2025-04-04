include ../../Makefile.in

default: meld

meld:
	@echo "Building MELD example for GPU"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 meld_gpu.cu ${LINK}

lin-meld:
	@echo "Building Linear MELD example for GPU"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 test_lin_meld.cu ${LINK}

run:
	make lin-meld
	./a.out
	paraview xa0.vtk