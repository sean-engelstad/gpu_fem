include ../../../Makefile.in
include ${TACS_DIR}/Makefile.in
include ${TACS_DIR}/TACS_Common.mk

default: kmat

kmat:
	echo "${TACS_DIR}/Makefile.in"
	rm cpu_kmat.txt || echo "no *.txt files to remove"
	make cpu_kmat.o
	${CXX} -o cpu_kmat.out cpu_kmat.o ${TACS_LD_FLAGS}
	./cpu_kmat.out >> cpu_kmat.txt

precond:
	rm cpu_precond.txt || echo "no *.txt files to remove"
	make cpu_precond.o
	${CXX} -o cpu_precond.out cpu_precond.o ${TACS_LD_FLAGS}
	./cpu_precond.out >> cpu_precond.txt

gmres:
	make cpu_gmres.o
	${CXX} -o cpu_gmres.out cpu_gmres.o ${TACS_LD_FLAGS}
	./cpu_gmres.out

out:
# writeout CPU precond matrix sparsity
	make cpu_out.o
	${CXX} -o cpu_out.out cpu_out.o ${TACS_LD_FLAGS}
	./cpu_out.out

clean:
	rm -f *.o
	rm -f *.x
	rm -f *.f5
	rm -f *.vtk
	rm -f *.out
