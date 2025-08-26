// class types for different coarse-fine or prolongation operators
// the restriction is often the transpose or row-normalized transpose for geom multigrid
#pragma once
#include "structured_plate.h"


class CylinderProlongation : public PlateProlongation {
    /* 
    this is a simpler version of a structured cylinder mesh with 2x refinement on hoop and axial direcs 
    there are nxe elems in axial and nxe in hoop (so that nnodes = nxe*(nxe+1) and nelems = nxe^2)

    // still the inode = (nxe+1) * iy + ix, etc..

    note then that nnodes = nxe*(nxe+1) which differs from plate (cause back edge connects)
    it actually uses the same structured mesh prolong code (though nelems is different that is arg),
    so same methods I think..
    */
};