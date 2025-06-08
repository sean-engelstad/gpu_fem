include ../../../Makefile.in

default: kmat

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}

kmat:
	rm gpu_kmat.txt || echo "no *.txt files to remove"
	make gpu_kmat 
	./gpu_kmat.out >> gpu_kmat.txt

precond:
	rm gpu_precond.txt || echo "no *.txt files to remove"
	make gpu_precond && ./gpu_precond.out >> gpu_precond.txt

gmres:
	rm gpu_gmres.txt || echo "no *.txt files to remove"
	make gpu_gmres && ./gpu_gmres.out >> gpu_gmres.txt

funcs:
	rm gpu_funcs.txt || echo "no *.txt files to remove"
	make gpu_funcs && ./gpu_funcs.out

cpu_precond:
	rm cpu_to_gpu_precond.txt || echo "no *.txt files to remove"
	make cpu_to_gpu_precond
	./cpu_to_gpu_precond.out >> cpu_to_gpu_precond.txt

cpu_gmres:
	rm cpu_to_gpu_gmres.txt || echo "no *.txt files to remove"
	make cpu_to_gpu_gmres
	./cpu_to_gpu_gmres.out

new:
	$(MAKE) clean
	$(MAKE) debug

clean:
	rm -f *.o
	rm -f *.x
	rm -f *.f5
	rm -f *.vtk
