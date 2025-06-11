include ../../Makefile.in

default: _write_reordering

%: %.cpp
	g++ ${INCLUDE_FLAGS} $< -o $@.out ${MPI_LIB} ${METIS_LIB}

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}

test:
	make clean
	make 5_mat
	./5_mat.out RCM nofill

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"