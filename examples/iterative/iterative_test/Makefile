include ../../../Makefile.in

default: gmres

gmres:
	${GPU_CXX} -g ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 gmres.cu ${LINK}

solve_Hyg:
	${GPU_CXX} -g ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 solve_Hyg_system.cu ${LINK}

solve_Hyg2:
	${GPU_CXX} -g ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 solve_Hyg_system_2x2.cu ${LINK}

reordering:
	${CXX} -g -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 reordering.cpp

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"