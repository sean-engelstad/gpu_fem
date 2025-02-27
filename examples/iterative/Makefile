include ../../Makefile.in

default: gmres

gmres:
	${GPU_CXX} -g ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 gmres.cu ${LINK}

reordering:
	${CXX} -g -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 reordering.cpp

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"