
CUDA_LINK := -lcublas -lcusparse -lcudart -lcusolver

csr: 1_gmres_csr

bsr: 2_gmres_bsr

%: %.cu
	nvcc $< -o $@.out ${CUDA_LINK}