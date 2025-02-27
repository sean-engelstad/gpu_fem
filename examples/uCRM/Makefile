include ../../Makefile.in

default: gpu

gpu:
	@echo "Building uCRM example for GPU"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 linear_static.cu ${LINK}
	# ${GPU_CXX} -g ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 linear_static.cu ${LINK}

run:
	rm out.txt || echo "no out.txt"
	rm a.out || echo "no a.out"
	make gpu
	./a.out >> out.txt

load_mesh:
	${CXX} ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 load_mesh.cpp ${CPU_LINK}