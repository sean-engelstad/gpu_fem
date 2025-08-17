include ../../Makefile.in

default: partial_derivs

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}

test:
	make partial_derivs
	./partial_derivs.out

clean:
	rm -f *.o
	rm -f *.x
	rm -f *.f5
	rm -f *.vtk
