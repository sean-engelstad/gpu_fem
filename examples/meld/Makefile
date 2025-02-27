include ../../Makefile.in

default: lin-meld

meld:
	@echo "Building Linear MELD example for GPU"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 test_meld.cu ${LINK}

lin-meld:
	@echo "Building MELD example for GPU"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 test_lin_meld.cu ${LINK}

run:
	make lin-meld
	./a.out
	paraview xa0.vtk