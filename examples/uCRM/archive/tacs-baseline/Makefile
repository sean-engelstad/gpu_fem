include ${TACS_DIR}/Makefile.in
include ${TACS_DIR}/TACS_Common.mk

default: tacs-baseline.o crm

crm:
	${CXX} -o crm.x tacs-baseline.o ${TACS_LD_FLAGS}

debug: TACS_CC_FLAGS=${TACS_DEBUG_CC_FLAGS}
debug: default

complex: TACS_DEF="-DTACS_USE_COMPLEX"
complex: default

complex_debug: TACS_DEF="-DTACS_USE_COMPLEX"
complex_debug: debug


new:
	$(MAKE) clean
	$(MAKE) debug

clean:
	rm -f *.o
	rm -f *.x
	rm -f *.f5
	rm -f *.vtk