include ../../Makefile.in

default: svd

svd:
	g++ ${INCLUDE_FLAGS} svd_3x3.cpp -o svd_3x3.out

eig:
	g++ ${INCLUDE_FLAGS} eig_3x3.cpp -o eig_3x3.out

eig-jac:
	g++ ${INCLUDE_FLAGS} eig_3x3_jac.cpp -o eig_3x3_jac.out

svd-jac:
	g++ ${INCLUDE_FLAGS} svd_3x3_jac.cpp -o svd_3x3_jac.out