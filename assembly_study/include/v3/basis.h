#pragma once
#include "_basis_utils.h"
#include "a2dcore.h"
#include "quadrature.h"

// only considering order = 2 here, can generalize later if I want..
template <typename T, class Quadrature_>
class ShellQuadBasisV3 {
   public:
    using Quadrature = Quadrature_;

    // order and number of nodes
    static constexpr int32_t order = 2;
    static constexpr int32_t num_nodes = 4;
    static constexpr int32_t num_tying = 9;

    // isoperimetric has same #nodes geometry class inside it
    class LinearQuadGeo {
       public:
        static constexpr int32_t spatial_dim = 3;
        static constexpr int32_t num_nodes = 4;
        static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;
    };  // end of class LinearQuadGeo
    using Geo = LinearQuadGeo;
};