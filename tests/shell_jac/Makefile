
TACS_DIR = ${HOME}/git/tacs

# GPU_FEM include
include ../../Makefile.in

# TACS include
include ${TACS_DIR}/Makefile.in
include ${TACS_DIR}/TACS_Common.mk

check-jac:
# check jac against TACS
	make clean
	make jac
	./jac-cpu.out >> out1.txt
	make jac-tacs
	./tacs-jac.x >> out2.txt
	diff out1.txt out2.txt >> diff.txt

check-jac-cs:
# check jacobian complex-step
	make clean
	make jac
	./jac-cpu.out >> out-anl.txt
	make jac-cs
	./jac-cs.out >> out-cs.txt
	diff out-anl.txt out-cs.txt >> diff.txt

check-jac-gpu:
# check jac CPU vs GPU
	make clean
	make jac
	./jac-cpu.out >> out-cpu.txt
	make jac-gpu
	./jac-gpu.out >> out-gpu.txt

jac:
    # builds analytic residual on CPU
	@echo "Building with CPU support"
	g++ -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 jac.cu -o jac-cpu.out

jac-gpu:
    # builds analytic residual on GPU
	@echo "Building with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${INCLUDE_FLAGS} -std=c++17 jac.cu -o jac-gpu.out

jac-cs:
    # complex-step jacobian
	@echo "Building with CPU support"
	g++ -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 jac-cs.cu -o jac-cs.out

jac-tacs: jac-tacs.o
	@echo "Building with TACS CPU"
	mpicxx -o tacs-jac.x jac-tacs.o ${TACS_LD_FLAGS}

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"
	rm *.o || echo "no .o files to remove"
	rm *.x || echo "no .x files to remove"

run:
	${MAKE} clean
	${MAKE} 2> make.txt
	./a.out