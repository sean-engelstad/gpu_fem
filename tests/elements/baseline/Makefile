TACS_DIR := ${HOME}/git/_archive/tacs

# GPU_FEM include
# include ../../../Makefile.in

# TACS include
include ${TACS_DIR}/Makefile.in
include ${TACS_DIR}/TACS_Common.mk

default: res-tacs

res-tacs: res-tacs.o
	${CXX} -o tacs-res.x res-tacs.o ${TACS_LD_FLAGS}

jac-tacs: jac-tacs.o
	${CXX} -o tacs-jac.x jac-tacs.o ${TACS_LD_FLAGS}

# res-tacs: res-tacs.o

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"
	rm *.o || echo "no .o files to remove"
	rm *.x || echo "no .x files to remove"