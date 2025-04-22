
# GPU_FEM include
include ../../Makefile.in

# TACS include
include ${TACS_DIR}/Makefile.in
include ${TACS_DIR}/TACS_Common.mk

check-jac:
# check jac against TACS
	make clean
	make jac-gpu
	./jac-gpu.out >> out1.txt
	make jac-tacs
	./tacs-jac.x >> out2.txt
	diff out1.txt out2.txt >> diff.txt

check-jacNL:
# check jac against TACS
	make clean
	make jac-gpuNL
	./jac-gpu.out >> out1.txt
	make jac-tacsNL
	./tacs-jac.x >> out2.txt
	diff out1.txt out2.txt >> diff.txt

jac-gpu:
    # builds analytic residual on GPU
	@echo "Building with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${INCLUDE_FLAGS} -std=c++17 jac.cu -o jac-gpu.out

jac-tacs: jac-tacs.o
	@echo "Building with TACS CPU"
	mpicxx -o tacs-jac.x jac-tacs.o ${TACS_LD_FLAGS}

jac-gpuNL:
    # builds analytic residual on GPU
	@echo "Building NL with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -DNLINEAR -Xcompiler ${INCLUDE_FLAGS} -std=c++17 jac.cu -o jac-gpu.out

jac-tacsNL: jac-tacs.o
	@echo "Building NL with TACS CPU"
	${CXX} -DNLINEAR ${TACS_CC_FLAGS} -std=c++11 -c jac-tacs.cpp -o jac-tacs.o
	mpicxx -o tacs-jac.x jac-tacs.o ${TACS_LD_FLAGS}

test-tying-strain:
	g++ ${INCLUDE_FLAGS} test-tying-strain.cpp

a2d-shell-strain:
	g++ ${INCLUDE_FLAGS} a2d_shellstrain.cpp

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"
	rm *.o || echo "no .o files to remove"
	rm *.x || echo "no .x files to remove"

run:
	${MAKE} clean
	${MAKE} 2> make.txt
	./a.out