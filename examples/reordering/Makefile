include ../../Makefile.in

default: reordering

reordering:
	${CXX} -g -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 reordering.cpp

gmres:
	${GPU_CXX} -g ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 gmres.cu ${LINK}

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"