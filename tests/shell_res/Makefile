
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

check-res:
# check residual against TACS
	make clean
	make res
	./res.out >> out1.txt
	make res-tacs
	./tacs-res.x >> out2.txt
	diff out1.txt out2.txt >> diff.txt

check-res3:
# check residual against TACS (all three methods)
	make clean
	make res
	./res.out >> out1.txt
	make res-jac
	./res-jac.out >> out2.txt
	make res-tacs
	./tacs-res.x >> out3.txt
# diff out1.txt out2.txt >> diff.txt

check-res-cs:
# check residual complex-step
	make clean
	make res
	./res.out >> out-anl.txt
	make res-cs
	./res-cs.out >> out-cs.txt
	diff out-anl.txt out-cs.txt >> diff.txt
	
check-gpu:
# check CPU and GPU give same answer
	make clean
	make res
	./res.out >> out-cpu.txt
	make res-gpu
	./res-gpu.out >> out-gpu.txt
	diff out-cpu.txt out-gpu.txt >> diff.txt

res:
    # builds analytic residual on CPU
	@echo "Building with CPU support"
	g++ -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 res.cu -o res.out

res-jac:
    # builds analytic residual on CPU
	@echo "Building with CPU support"
	g++ -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 res-jac.cu -o res-jac.out

res-gpu:
    # builds analytic residual on GPU
	@echo "Building with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${INCLUDE_FLAGS} -std=c++17 res.cu -o res-gpu.out

res-cs:
    # complex-step residual
	@echo "Building with CPU support"
	g++ -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 res_cs.cu -o res-cs.out

res-tacs: res-tacs.o
	@echo "Building with TACS CPU"
	${CXX} -o tacs-res.x res-tacs.o ${TACS_LD_FLAGS}

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"
	rm *.o || echo "no .o files to remove"
	rm *.x || echo "no .x files to remove"

run:
	${MAKE} clean
	${MAKE} 2> make.txt
	./a.out