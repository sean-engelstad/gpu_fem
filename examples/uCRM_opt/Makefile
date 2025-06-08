include ../../Makefile.in

PY_EXT = $(shell python3-config --extension-suffix)

default: 
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -shared -Xcompiler "-fPIC -Wno-deprecated-declarations" -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} ${PYBIND11_INCLUDE} -std=c++17 binding.cu -o gpusolver$(PY_EXT) ${LINK}	


clean:
	rm *.so
	rm -f *.x
	rm -f *.f5
	rm -f *.vtk
	rm -f out/*.vtk
