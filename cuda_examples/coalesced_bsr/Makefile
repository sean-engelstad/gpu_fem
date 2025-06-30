
CUDA_LINK := -lcublas -lcusparse -lcudart -lcusolver

write: coalesced_write
	./coalesced_write.out

%: %.cu
	nvcc -Xcompiler "-Wno-deprecated-declarations" $< -o $@.out ${CUDA_LINK}