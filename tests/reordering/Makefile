include ../../Makefile.in

default: _write_reordering

%: %.cpp
	g++ ${INCLUDE_FLAGS} $< -o $@.out

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"