include ../../Makefile.in

TACS_DIR = ${HOME}/git/tacs
TACS_INCLUDE = ${TACS_DIR}/include

default:
	@if [ "$(USE_GPU)" = "1" ]; then \
	    $(MAKE) gpu; \
	else \
	    $(MAKE) cpu; \
	fi

gpu:
	@echo "Building with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${INCLUDE_FLAGS} -std=c++17 main.cu

cpu:
	@echo "Building with CPU support"
	${CXX} -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 main.cu

tacs:
	@echo "Building tacs verification example"
	${CXX} ${CC_FLAGS} ${TACS_INCLUDE} tacs-verify.cpp

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"

run:
	${MAKE} clean
	${MAKE} 2> make.txt
	./a.out