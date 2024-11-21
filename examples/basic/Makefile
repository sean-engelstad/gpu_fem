
A2D_DIR := ${HOME}/git/a2d
A2D_INCLUDE := -I${A2D_DIR}/include
# USE_GPU := 1
USE_GPU := 0
GPU_CXX = nvcc
CXX = g++
CC_FLAGS = -g
GPU_CC_FLAGS := #-w

default:
	@if [ "$(USE_GPU)" = "1" ]; then \
	    $(MAKE) gpu; \
	else \
	    $(MAKE) cpu; \
	fi

gpu:
	@echo "Building with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${A2D_INCLUDE} -std=c++17 main.cu

cpu:
	@echo "Building with CPU support"
	${CXX} -x c++ ${CC_FLAGS} ${A2D_INCLUDE} -std=c++17 main.cu

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"

run:
	${MAKE} clean
	${MAKE} 2> make.txt
	./a.out