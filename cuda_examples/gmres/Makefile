
CUDA_LINK := -lcublas -lcusparse -lcudart -lcusolver

csr: 1_gmres_csr

bsr: 2_gmres_bsr

csr2: 
	make csr && ./1_gmres_csr.out

bsr2: 
	make bsr && ./2_gmres_bsr.out

%: %.cu
	nvcc -Xcompiler "-Wno-deprecated-declarations" $< -o $@.out ${CUDA_LINK}