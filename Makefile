
A2D_DIR := ${HOME}/git/a2d
A2D_INCLUDE := -I${A2D_DIR}/include

default:
	nvcc -w -DUSE_GPU -Xcompiler ${A2D_INCLUDE} -std=c++17 main.cu
clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"

run:
	${MAKE} clean
	${MAKE} 2> make.txt
	./a.out