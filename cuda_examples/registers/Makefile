
CUDA_LINK := -lcublas -lcusparse -lcudart -lcusolver

A2D_DIR := ${HOME}/git/_archive/a2d
A2D_INCLUDE := -I${A2D_DIR}/include

%: %.cu
	nvcc -Xcompiler "-Wno-deprecated-declarations" -Xcompiler ${A2D_INCLUDE} $< -o $@.out ${CUDA_LINK}

profile: a2d_matmul
	rm -rf report.ncu-rep || echo "no previous nsight compute report.ncu-rep files"
	sudo /usr/local/cuda-12.6/bin/ncu --set full --export  report.ncu-rep ./a2d_matmul.out
	ncu-ui report.ncu-rep