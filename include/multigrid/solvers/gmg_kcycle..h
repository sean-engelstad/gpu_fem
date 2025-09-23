#pragma once

#include "gmg.h"

template <class GRID, class KrylovSolver>
class KrylovGeometricMultigridSolver : public GeometricMultigridSolver<GRID> {


    

public: // private
    std::vector<KrylovSolver> solvers;
};