include ../../Makefile.in

default: svd

svd:
	g++ ${INCLUDE_FLAGS} svd.cpp -o svd.out

eig:
	g++ ${INCLUDE_FLAGS} eig.cpp -o eig.out

eig-jac:
	g++ ${INCLUDE_FLAGS} eig_jac.cpp -o eig_jac.out

rot-jac:
	g++ ${INCLUDE_FLAGS} rot_jac.cpp -o rot_jac.out

clean:
	rm *.out || echo "no *.out files"