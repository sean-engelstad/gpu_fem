include ../../Makefile.in

CUDA_LINK := -lcublas -lcusparse -lcudart -lcusolver

scan: _scan

clean:
	rm *out || echo "no *.out files"

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}