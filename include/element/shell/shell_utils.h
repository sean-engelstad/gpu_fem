#pragma once
#include "../../cuda_utils.h"
#include "_basis_utils.h"

// for now still include the old shell utils methods
#include "shell_utils_slow.h"
// below are all the newer fast ones on GPU

template <typename T, class Basis, class Data>
__HOST_DEVICE__ void ShellComputeTransformFast(const T refAxis[], const T &xi, const T &eta,
                                               const T xpts[], const T n0[], T Tmat[]) {
    // make the normal a unit vector, store it in Tmat (Tmat assembled in transpose form first for
    // convenience, then transposed later)
    for (int i = 0; i < 9; i++) Tmat[i] = 0.0;  // zero out
    for (int i = 0; i < 3; i++) {
        Tmat[6 + i] = n0[i];
    }
    T *n = &Tmat[6];
    T norm = sqrt(A2D::VecDotCore<T, 3>(n, n));
    for (int i = 0; i < 3; i++) {
        n[i] /= norm;
    }

    // set t1
    if constexpr (Data::has_ref_axis) {
        // shell ref axis transform
        A2D::VecAddCore<T, 3>(1.0, refAxis, &Tmat[0]);
    } else {  // doesn't have ref axis
        // shell natural transform, set to dX/dxi
        for (int i = 0; i < 3; i++) {
            Tmat[i] = Basis::template interpFieldsGradFast<XI, 3>(i, xi, eta, xpts);
        }
    }

    // remove normal component from t1 (of n)
    T d = A2D::VecDotCore<T, 3>(&Tmat[0], &Tmat[6]);  // t1 cross n
    A2D::VecAddCore<T, 3>(T(1.0), &Tmat[0], &Tmat[3]);
    A2D::VecAddCore<T, 3>(-d, &Tmat[6], &Tmat[3]);  // t1 - (t1 dot n) * n => t2 (temp store in t2)
    norm = sqrt(A2D::VecDotCore<T, 3>(&Tmat[3], &Tmat[3]));
    A2D::VecAddCore<T, 3>(1.0, &Tmat[3], &Tmat[0]);

    // compute t2 by cross product
    A2D::VecCrossCore<T>(&Tmat[6], &Tmat[0], &Tmat[3]);  // nhat cross t1hat => t2hat

    // transpose Tmat to standard column major from row major format
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            // in place transpose swap
            T tmp = Tmat[3 * i + j];
            Tmat[3 * i + j] = Tmat[3 * j + i];
            Tmat[3 * j + i] = tmp;
        }
    }
}

template <typename T, class Data, class Basis>
__HOST_DEVICE__ void ShellComputeNodalTransforms(const int inode, const T xpts[],
                                                 const Data physData, T Tmat[9], T XdinvT[9]) {
    T pt[2];
    Basis::getNodePoint(inode, pt);

    // get shell transform and Xdn frame scope
    T Xdinv[9];
    {
        T n0[3];
        Basis::ShellComputeNodeNormalLight(pt, xpts, n0);

        // assemble Xd frame (Tmat treated here as Xd)
        T *Xd = &Tmat[0];
        Basis::assembleFrameLight<3>(pt, xpts, n0, Xd);
        A2D::MatInvCore<T, 3>(Xd, Xdinv);

        // compute the shell transform based on the ref axis in Data object
        ShellComputeTransformLight<T, Basis, Data>(physData.refAxis, pt, xpts, n0, Tmat);
    }  // end of Xd and shell transform scope

    // get full transform product
    A2D::MatMatMultCore3x3<T>(Xdinv, Tmat, XdinvT);
}

template <typename T, class Data, class Basis, class Quadrature>
__HOST_DEVICE__ void ShellComputeQuadptTransforms(const int iquad, const T xpts[],
                                                  const Data physData, T Tmat[9], T XdinvT[9]) {
    T pt[2];
    Quadrature::getQuadraturePoint(iquad, pt);

    // get shell transform and Xdn frame scope
    T Xdinv[9];
    {
        T n0[3];
        Basis::ShellComputeNodeNormalLight(pt, xpts, n0);

        // assemble Xd frame (Tmat treated here as Xd)
        T *Xd = &Tmat[0];
        Basis::assembleFrameLight<3>(pt, xpts, n0, Xd);
        A2D::MatInvCore<T, 3>(Xd, Xdinv);

        // compute the shell transform based on the ref axis in Data object
        ShellComputeTransformLight<T, Basis, Data>(physData.refAxis, pt, xpts, n0, Tmat);
    }  // end of Xd and shell transform scope

    // get full transform product
    A2D::MatMatMultCore3x3<T>(Xdinv, Tmat, XdinvT);
}

template <typename T, int vars_per_node, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainFast(const T quad_pt[], const T xpts[], const T vars[],
                                                 const T Tmat[], const T XdinvT[], T &et) {
    // instead of storing etn[4], we add to interpolated et on the fly..
    et = 0.0;
    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        T pt[2];
        Basis::getNodePoint(inode, pt);

        // const T Tmat = &sTmat[9 * inode];
        // const T XdinvT = &sXdinvT[9 * inode];

        // // assemble u0xn frame scope
        T u0xn[9], zero[3];
        {
            for (int i = 0; i < 3; i++) zero[i] = 0.0;
            Basis::assembleFrameLight<6>(pt, vars, zero, u0xn);
            for (int i = 0; i < 3; i++)
                u0xn[3 * i + 2] = 0.0;  // last column is zero (just put Xd in
            // there to save registers)
        }

        // compute rotation matrix at this node
        T C[9], tmp[9];
        Director::template computeRotationMat<vars_per_node, 1>(&vars[vars_per_node * inode], C);

        // compute Ct = T^T * C * T
        using MatOp = A2D::MatOp;
        A2D::MatMatMultCore3x3<T, MatOp::TRANSPOSE>(&Tmat[9 * inode], C, tmp);
        A2D::MatMatMultCore3x3<T>(tmp, &Tmat[9 * inode], C);

        // Compute transformation u0x = T^T * u0xn * (Xdinv*T)
        A2D::MatMatMultCore3x3<T>(u0xn, &XdinvT[9 * inode], tmp);
        A2D::MatMatMultCore3x3<T, MatOp::TRANSPOSE>(&Tmat[9 * inode], tmp, u0xn);

        // compute the drill strain
        T etn = Director::evalDrillStrain(u0xn, C);

        et += etn * Basis::lagrangeLobatto2DLight(inode, quad_pt[0], quad_pt[1]);
    }  // end of node for loop

}  // end of method ShellComputeDrillStrainFast

template <typename T, int vars_per_node, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainSensFast(const T quad_pt[], const T xpts[],
                                                     const T vars[], const T Tmat[],
                                                     const T XdinvT[], const T et_bar[], T res[]) {
    // TODO : do we actually need Ctn, Tn, XdinvTn, u0xn here?

    // first interpolate back to nodal level
    A2D::Vec<T, Basis::num_nodes> etn_bar;
    Basis::template interpFieldsTransposeLight<1, 1>(quad_pt, et_bar, etn_bar.get_data());

    constexpr A2D::MatOp NORM = A2D::MatOp::NORMAL;
    constexpr A2D::MatOp TRANS = A2D::MatOp::TRANSPOSE;

    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        T pt[2];
        Basis::getNodePoint(inode, pt);

        T u0xn_bar[9], C_bar[9];  // really u0x_bar at this point (but don't want two vars for it)
        Director::evalDrillStrainSens(etn_bar[inode], u0xn_bar, C_bar);

        T tmp[9];
        {
            // C_bar = T * C2_bar * T^t (in reverse)
            A2D::MatMatMultCore3x3<T>(&Tmat[9 * inode], C_bar, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, &Tmat[9 * inode], C_bar);

            // reverse C(vars) to C_bar => res
            Director::template computeRotationMatSens<vars_per_node, 1>(
                C_bar, &res[vars_per_node * inode]);
        }

        // backprop u0x_bar to u0xn_bar^T
        {
            // u0xn_bar = T * u0x_bar * XdinvT^t
            // transpose version for convenience of cols avail in rows
            // u0xn_bar^t = XdinvT * u0x_bar^t * T^t (u0xn_bar now holds transpose u0xn_bar^t)
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(&XdinvT[9 * inode], u0xn_bar, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, &Tmat[9 * inode], u0xn_bar);

            // reverse the interpolations u0xn_bar to res
            // because we have u0xn_bar^T stored, each row is u0xi_bar, u0eta_bar
            Basis::template interpFieldsGradTransposeLight<vars_per_node, 3>(pt, &u0xn_bar[0],
                                                                             &u0xn_bar[3], res);
        }

    }  // end of node for loop
}  // end of method ShellComputeDrillStrainSensFast

template <typename T, class Physics, class Basis, class Director>
__HOST_DEVICE__ static void computeInterpTyingStrainFast(const T quad_pt[], const T xpts[],
                                                         const T vars[], T gty[]) {
    // using unrolled loop here for efficiency (if statements and for loops not
    // great for device)
    // int32_t offset, num_tying;
    static constexpr bool is_nonlinear = Physics::is_nonlinear;
    static constexpr int vars_per_node = Physics::vars_per_node;
    static constexpr int order = Basis::order;

    T tying_pt[2], ety, d0[3], n0[3];

    // if statements by order allows us to inline and manually unroll the loops given the order,
    // which reduces registers (instead of for loops which may not
    // go to registers if array indexing in for loops

    if constexpr (order == 2) {
        // g11 tying strain = X,xi * U0,xi + 0.5 * U0,xi * U0,xi
        // -----------------------------------------------------
#pragma unroll
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = 0.0;
            tying_pt[1] = -1 + itying * 2;
            ety =
                Basis::interpFieldsGradDotLight<XI, XI, 3, vars_per_node, 3>(tying_pt, xpts, vars);
            if constexpr (is_nonlinear)
                ety +=
                    0.5 * Basis::interpFieldsGradDotLight<XI, XI, vars_per_node, vars_per_node, 3>(
                              tying_pt, vars, vars);
            gty[0] += ety * Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[1]);
        }

        // g12 tying strain = 0.5 * (X,eta * U0,xi + X,xi * U0,eta) + 0.5 * U0,xi * U0,eta
        // -----------------------------------------------------
#pragma unroll
        for (int itying = 0; itying < 1; itying++) {
            tying_pt[0] = 0.0;
            tying_pt[1] = 0.0;
            Basis::template getTyingPoint<2>(itying, tying_pt);
            ety = 0.5 * Basis::interpFieldsGradDotLight<ETA, XI, 3, vars_per_node, 3>(tying_pt,
                                                                                      xpts, vars);
            ety += 0.5 * Basis::interpFieldsGradDotLight<XI, ETA, 3, vars_per_node, 3>(tying_pt,
                                                                                       xpts, vars);
            if constexpr (is_nonlinear)
                ety +=
                    0.5 * Basis::interpFieldsGradDotLight<XI, ETA, vars_per_node, vars_per_node, 3>(
                              tying_pt, vars, vars);
            gty[1] += ety;  // only one tying ponit, just becomes the quadpt (MITC)
        }

        // g13 = 0.5 * (X,xi * d0 + U,xi * n0) + 0.5 * U,xi * d0
        // -----------------------------------------------------
#pragma unroll
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = 0.0;
            tying_pt[1] = -1 + itying * 2;

            Director::template interpDirectorLight<Basis, vars_per_node, Basis::num_nodes>(
                tying_pt, xpts, vars, d0);
            ety = 0.5 * Basis::interpFieldsGradRightDotLight<XI, 3, 3>(tying_pt, xpts, d0);
            {
                Basis::interpNodeNormalLight(tying_pt, xpts, n0);
                ety += 0.5 * Basis::interpFieldsGradRightDotLight<XI, vars_per_node, 3>(tying_pt,
                                                                                        vars, n0);
            }

            if constexpr (is_nonlinear)
                ety += 0.5 * Basis::interpFieldsGradRightDotLight<XI, vars_per_node, 3>(tying_pt,
                                                                                        vars, d0);
            gty[2] += ety * Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[1]);
        }

        // g22 tying strain = X,eta * U0,eta + 0.5 * U0,eta * U0,eta
        // -----------------------------------------------------
#pragma unroll
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = -1 + itying * 2;
            tying_pt[1] = 0.0;
            ety = Basis::interpFieldsGradDotLight<ETA, ETA, 3, vars_per_node, 3>(tying_pt, xpts,
                                                                                 vars);
            if constexpr (is_nonlinear)
                ety += 0.5 *
                       Basis::interpFieldsGradDotLight<ETA, ETA, vars_per_node, vars_per_node, 3>(
                           tying_pt, vars, vars);
            gty[3] += ety * Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[0]);
        }

        // g23 tying strain = X,eta * U0,eta + 0.5 * U0,eta * U0,eta
        // -----------------------------------------------------
#pragma unroll
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = -1 + itying * 2;
            tying_pt[1] = 0.0;

            Director::template interpDirectorLight<Basis, vars_per_node, Basis::num_nodes>(
                tying_pt, xpts, vars, d0);
            ety = 0.5 * Basis::interpFieldsGradRightDotLight<ETA, 3, 3>(tying_pt, xpts, d0);
            {
                Basis::interpNodeNormalLight(tying_pt, xpts, n0);
                ety += 0.5 * Basis::interpFieldsGradRightDotLight<ETA, vars_per_node, 3>(tying_pt,
                                                                                         vars, n0);
            }

            if constexpr (is_nonlinear)
                ety += 0.5 * Basis::interpFieldsGradRightDotLight<ETA, vars_per_node, 3>(tying_pt,
                                                                                         vars, d0);
            gty[4] += ety * Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[0]);
        }

        // g33 = 0, repr gty[5]
    }

    // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
    //     printf("gty = %.4e, %.4e, %.4e, %.4e, %.4e\n", gty[0], gty[1], gty[2], gty[3], gty[4]);
}  // end of computeInterpTyingStrainFast

template <typename T, class Physics, class Basis, class Director>
__HOST_DEVICE__ static void computeInterpTyingStrainFastSens(const T quad_pt[], const T xpts[],
                                                             const T vars[], const T gty_bar[],
                                                             T res[]) {
    // using unrolled loop here for efficiency (if statements and for loops not
    // great for device)
    // int32_t offset, num_tying;
    static constexpr bool is_nonlinear = Physics::is_nonlinear;
    static constexpr int vars_per_node = Physics::vars_per_node;
    static constexpr int order = Basis::order;

    T tying_pt[2], ety_bar, d0[3], n0[3];

    // if statements by order allows us to inline and manually unroll the loops given the order,
    // which reduces registers (instead of for loops which may not
    // go to registers if array indexing in for loops

    if constexpr (order == 2) {
        // g11 tying strain = X,xi * U0,xi + 0.5 * U0,xi * U0,xi
        // -----------------------------------------------------
#pragma unroll
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = 0.0;
            tying_pt[1] = -1 + itying * 2;

            ety_bar =
                gty_bar[0] * Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[1]);

            Basis::template addInterpFieldsGradDotSensLight<XI, XI, 3, vars_per_node, 3>(
                ety_bar, tying_pt, xpts, res);

            if constexpr (is_nonlinear)
                Basis::template addInterpFieldsGradDotSensLight<XI, XI, vars_per_node,
                                                                vars_per_node, 3>(ety_bar, tying_pt,
                                                                                  vars, res);
        }

        // g12 tying strain = 0.5 * (X,eta * U0,xi + X,xi * U0,eta) + 0.5 * U0,xi * U0,eta
        // -----------------------------------------------------
#pragma unroll
        for (int itying = 0; itying < 1; itying++) {
            tying_pt[0] = 0.0;
            tying_pt[1] = 0.0;

            ety_bar = 0.5 * gty_bar[1];
            Basis::template addInterpFieldsGradDotSensLight<ETA, XI, 3, vars_per_node, 3>(
                ety_bar, tying_pt, xpts, res);
            Basis::template addInterpFieldsGradDotSensLight<XI, ETA, 3, vars_per_node, 3>(
                ety_bar, tying_pt, xpts, res);

            if constexpr (is_nonlinear) {
                Basis::template addInterpFieldsGradDotSensLight<XI, ETA, vars_per_node,
                                                                vars_per_node, 3>(ety_bar, tying_pt,
                                                                                  vars, res);
                Basis::template addInterpFieldsGradDotSensLight<ETA, XI, vars_per_node,
                                                                vars_per_node, 3>(ety_bar, tying_pt,
                                                                                  vars, res);
            }
        }

        // g13
#pragma unroll
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = 0.0;
            tying_pt[1] = -1 + itying * 2;

            ety_bar = 0.5 * gty_bar[2] *
                      Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[1]);

            A2D::Vec<T, 3> d0_bar;
            Basis::template interpFieldsGradRightDotLight_RightSens<XI, 3, 3>(
                ety_bar, tying_pt, xpts, d0_bar.get_data());
            Basis::template interpNodeNormalLight(tying_pt, xpts, n0);
            Basis::template interpFieldsGradRightDotLight_LeftSens<XI, vars_per_node, 3>(
                ety_bar, tying_pt, res, n0);

            // nonlinear g13 strain term += 0.5 * U,eta dot d0
            if constexpr (is_nonlinear) {
                Director::template interpDirectorLight<Basis, vars_per_node, Basis::num_nodes>(
                    tying_pt, xpts, vars, d0);

                Basis::template interpFieldsGradRightDotLight_LeftSens<XI, vars_per_node, 3>(
                    ety_bar, tying_pt, res, d0);
                Basis::template interpFieldsGradRightDotLight_RightSens<XI, vars_per_node, 3>(
                    ety_bar, tying_pt, vars, d0_bar.get_data());
            }
            Director::template interpDirectorLightSens<Basis, vars_per_node, Basis::num_nodes>(
                1.0, tying_pt, xpts, d0_bar.get_data(), res);
        }

        // g22
#pragma unroll  // for low num_tying can speed up?
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = -1 + itying * 2;
            tying_pt[1] = 0.0;

            ety_bar =
                gty_bar[3] * Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[0]);

            Basis::template addInterpFieldsGradDotSensLight<ETA, ETA, 3, vars_per_node, 3>(
                ety_bar, tying_pt, xpts, res);

            if constexpr (is_nonlinear) {
                // backprop g22 nl term 1/2 * U0,eta dot U0,eta
                Basis::template addInterpFieldsGradDotSensLight<ETA, ETA, vars_per_node,
                                                                vars_per_node, 3>(ety_bar, tying_pt,
                                                                                  vars, res);
            }
        }  // end of itying for loop for g22

        // g23
#pragma unroll
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = -1 + itying * 2;
            tying_pt[1] = 0.0;

            ety_bar = 0.5 * gty_bar[4] *
                      Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[0]);

            A2D::Vec<T, 3> d0_bar;
            Basis::template interpFieldsGradRightDotLight_RightSens<ETA, 3, 3>(
                ety_bar, tying_pt, xpts, d0_bar.get_data());
            Basis::template interpNodeNormalLight(tying_pt, xpts, n0);
            Basis::template interpFieldsGradRightDotLight_LeftSens<ETA, vars_per_node, 3>(
                ety_bar, tying_pt, res, n0);

            // nonlinear g23 strain term += 0.5 * U,eta dot d0
            if constexpr (is_nonlinear) {
                Director::template interpDirectorLight<Basis, vars_per_node, Basis::num_nodes>(
                    tying_pt, xpts, vars, d0);

                Basis::template interpFieldsGradRightDotLight_LeftSens<ETA, vars_per_node, 3>(
                    ety_bar, tying_pt, res, d0);
                Basis::template interpFieldsGradRightDotLight_RightSens<ETA, vars_per_node, 3>(
                    ety_bar, tying_pt, vars, d0_bar.get_data());
            }
            Director::template interpDirectorLightSens<Basis, vars_per_node, Basis::num_nodes>(
                1.0, tying_pt, xpts, d0_bar.get_data(), res);
        }

        // g33 = 0
    }
}  // end of computeInterpTyingStrainFastSens
