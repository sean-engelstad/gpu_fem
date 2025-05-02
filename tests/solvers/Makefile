include ../../Makefile.in

default: gmres_laplace

plate:
	make gmres_plate
	make LU_plate

clean:
	rm *.out || echo "no *.out files to remove"

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}