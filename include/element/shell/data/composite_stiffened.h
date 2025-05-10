#pragma once
#include "../../../cuda_utils.h"
#include "a2dcore.h"

// could be an internal class for each type of physics
// but I would like physics to be linear vs nonlinear and this class has methods which get ABD, etc.
template <typename T, bool has_ref_axis_ = true>
<<<<<<<< HEAD:include/element/shell/data/composite_stiffened.h
class ShellCompositeStiffenedData {
========
class ShellIsotropicData {
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:include/element/shell/data/isotropic.h
   public:
    static constexpr T transverseShearCorrectionFactor = 5.0 / 6.0;
    static constexpr T drillingRegularization = 10.0;  // default drilling regularization

<<<<<<<< HEAD:include/element/shell/data/composite_stiffened.h
    ShellCompositeStiffenedPanelData() = default;
========
    ShellIsotropicData() = default;
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:include/element/shell/data/isotropic.h
    static constexpr bool has_ref_axis = has_ref_axis_;

    // constructor with ref Axis
    template <bool U = has_ref_axis, typename std::enable_if<U, int>::type = 0>
<<<<<<<< HEAD:include/element/shell/data/composite_stiffened.h
    __HOST_DEVICE__ ShellCompositeStiffenedPanelData(T E_, T nu_, T thick_, T refAxis_[],
                                                     T tOffset_ = 0.0)
        : E(E_), nu(nu_), thick(thick_), tOffset(tOffset_) {
========
    __HOST_DEVICE__ ShellIsotropicData(T E_, T nu_, T thick_, T refAxis_[], T ys_ = 1.0,
                                       T rho_ = 1.0, T tOffset_ = 0.0)
        : E(E_), nu(nu_), thick(thick_), ys(ys_), rho(rho_), tOffset(tOffset_) {
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:include/element/shell/data/isotropic.h
        // deep copy the ref axis
        for (int i = 0; i < 3; i++) {
            refAxis[i] = refAxis_[i];
        }
    }

    // constructor without refAxis input
    template <bool U = has_ref_axis, typename std::enable_if<!U, int>::type = 0>
<<<<<<<< HEAD:include/element/shell/data/composite_stiffened.h
    __HOST_DEVICE__ ShellCompositeStiffenedPanelData(T E_, T nu_, T thick_, T tOffset_ = 0.0)
        : E(E_), nu(nu_), thick(thick_), tOffset(tOffset_) {}
========
    __HOST_DEVICE__ ShellIsotropicData(T E_, T nu_, T thick_, T ys_ = 1.0, T rho_ = 1.0,
                                       T tOffset_ = 0.0)
        : E(E_), nu(nu_), thick(thick_), ys(ys_), rho(rho_), tOffset(tOffset_) {}
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:include/element/shell/data/isotropic.h

    // constitutive methods
    __HOST_DEVICE__ static void evalTangentStiffness2D(const T E_, const T nu_, T C[]) {
        // isotropic C matrix for stress = C * strain in-plane with no twist
        T D = E_ / (1.0 - nu_ * nu_);
        C[0] = D;
        C[1] = nu_ * D;
        C[2] = 0.0;
        C[3] = D;
        C[4] = 0.0;
        T G = E_ / 2.0 / (1.0 + nu_);
        C[5] = G;
    }

    // TODO : add thickness offset (rarely used but to be complete.. can allow B matrix to be
    // nonzero)
<<<<<<<< HEAD:include/element/shell/data/composite_stiffened.h
    T E, nu, thick, tOffset;
========
    T E, nu, thick, tOffset, ys, rho;
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:include/element/shell/data/isotropic.h
    T refAxis[3];
};