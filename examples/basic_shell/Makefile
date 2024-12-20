include ../../Makefile.in

default:
	@if [ "$(USE_GPU)" = "1" ]; then \
	    $(MAKE) gpu; \
	else \
	    $(MAKE) cpu; \
	fi

res-cpu:
	@echo "Building with CPU support"
	${CXX} -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 res-cpu.cpp

res-gpu:
	@echo "Building with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${INCLUDE_FLAGS} -std=c++17 res-gpu.cu


jac-cpu:
	@echo "Building with CPU support"
	${CXX} -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 jac-cpu.cpp

jac-gpu:
	@echo "Building with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${INCLUDE_FLAGS} -std=c++17 jac-gpu.cu

test-gpu:
	@echo "Building with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${INCLUDE_FLAGS} -std=c++17 test_gpu.cu

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"

run:
	${MAKE} clean
	${MAKE} 2> make.txt
	./a.out