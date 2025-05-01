include ../../Makefile.in

default: reordering

reordering:
	${CXX} -g -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 reordering.cpp

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"