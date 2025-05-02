#TACS_DIR := ${HOME}/git/archive/tacs

# GPU_FEM include
include ../../../Makefile.in

# TACS include
include ${TACS_DIR}/Makefile.in
include ${TACS_DIR}/TACS_Common.mk

default: res-tacs

%: %.cpp %.o
	${CXX} -o tacs-res.x res-tacs.o ${TACS_LD_FLAGS}
	g++ ${INCLUDE_FLAGS} $< -o $@.out


clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"
	rm *.o || echo "no .o files to remove"
	rm *.x || echo "no .x files to remove"