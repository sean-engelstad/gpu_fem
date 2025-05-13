
# GPU_FEM include
include ../../Makefile.in

default: res-gpu

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}
	
%: %.cpp
	g++ ${INCLUDE_FLAGS} $< -o $@.out

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"
	rm *.o || echo "no .o files to remove"
	rm *.x || echo "no .x files to remove"

run:
	${MAKE} clean
	${MAKE} 2> make.txt
	./a.out