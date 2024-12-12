
TACS_DIR = ${HOME}/git/tacs

# GPU_FEM include
include ../../Makefile.in

# TACS include
include ${TACS_DIR}/Makefile.in
include ${TACS_DIR}/TACS_Common.mk

check-energy:
	make clean
	make energy
	./a.out >> out1.txt
	make energy-tacs
	./tacs-energy.x >> out2.txt
	diff out1.txt out2.txt >> diff.txt

energy:
	@echo "Building with CPU support"
	g++ -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 energy.cu

energy-tacs: energy-tacs.o
	@echo "Building with TACS CPU"
	${CXX} -o tacs-energy.x energy-tacs.o ${TACS_LD_FLAGS}

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"
	rm *.o || echo "no .o files to remove"
	rm *.x || echo "no .x files to remove"

run:
	${MAKE} clean
	${MAKE} 2> make.txt
	./a.out