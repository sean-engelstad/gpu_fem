include ../../Makefile.in

default: svd3x3

%: %.cpp
	g++ ${INCLUDE_FLAGS} $< -o $@.out

clean:
	rm *.out || echo "no *.out files"