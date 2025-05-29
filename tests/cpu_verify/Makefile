include ../../Makefile.in

export TACS_DIR=${HOME}/git/tacs/
include ${TACS_DIR}/Makefile.in
include ${TACS_DIR}/TACS_Common.mk

default: cpu_kmat.o cpu_kmat

cpu_kmat:
	echo "${TACS_LD_FLAGS}"
	${CXX} -o cpu_kmat.out cpu_kmat.o ${TACS_LD_FLAGS}

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}

mat_compare:
	rm *.txt || echo "no *.txt files to remove"
	make cpu_kmat.o
	make crm
	./crm.x >> cpu_kmat.txt
	make gpu_kmat
	./gpu_kmat.out >> gpu_kmat.txt


new:
	$(MAKE) clean
	$(MAKE) debug

clean:
	rm -f *.o
	rm -f *.x
	rm -f *.f5
	rm -f *.vtk
