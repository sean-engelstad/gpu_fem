#pragma once
#include "../../cuda_utils.h"
#include "_basis_utils.h"

template <typename T, class Data>
__HOST_DEVICE__ void ShellComputeTransform(const T refAxis[], const T dXdxi[], const T dXdeta[],
                                           const T n0[], T Tmat[]) {
    A2D::Vec<T, 3> n(n0), nhat;
    A2D::VecNormalize(n, nhat);

    A2D::Vec<T, 3> t1;
    if constexpr (Data::has_ref_axis) {
        // shell ref axis transform
        t1 = A2D::Vec<T, 3>(refAxis);
    } else {  // doesn't have ref axis
        // shell natural transform
        t1 = A2D::Vec<T, 3>(dXdxi);
    }

    // remove normal component from t1
    A2D::Vec<T, 3> temp, t1hat;
    T d = A2D::VecDotCore<T, 3>(nhat.get_data(), t1.get_data());
    A2D::VecSum(T(1.0), t1, -d, nhat, temp);
    A2D::VecNormalize(temp, t1hat);

    // compute t2 by cross product of normalized unit vectors
    A2D::Vec<T, 3> t2hat;
    A2D::VecCross(nhat, t1hat, t2hat);

    // save values in Tmat 3x3 matrix
    for (int i = 0; i < 3; i++) {
        Tmat[3 * i] = t1hat[i];
        Tmat[3 * i + 1] = t2hat[i];
        Tmat[3 * i + 2] = nhat[i];
    }
}

template <typename T, class Basis, class Data>
__HOST_DEVICE__ void ShellComputeTransformLight(const T refAxis[], const T pt[], const T xpts[],
                                           const T n0[], T Tmat[]) {
    // make the normal a unit vector, store it in Tmat (Tmat assembled in transpose form first for convenience, then transposed later)
    for (int i = 0; i < 9; i++) Tmat[i] = 0.0; // zero out
    for (int i = 0; i < 3; i++) {
        Tmat[6+i] = n0[i];
    }
    T *n = &Tmat[6];
    T norm = sqrt(A2D::VecDotCore<T,3>(n, n));
    for (int i = 0; i < 3; i++) {
        n[i] /= norm;
    }

    // set t1
    if constexpr (Data::has_ref_axis) {
        // shell ref axis transform
        A2D::VecAddCore<T,3>(1.0, refAxis, &Tmat[0]);
    } else {  // doesn't have ref axis
        // shell natural transform, set to dX/dxi
        for (int i = 0; i < 3; i++) {
            Tmat[i] = Basis::template interpFieldsGradLight<XI,3>(i, pt, xpts);
        }
    }

    // remove normal component from t1 (of n)
    T d = A2D::VecDotCore<T,3>(&Tmat[0], &Tmat[6]); // t1 cross n
    A2D::VecAddCore<T,3>(T(1.0), &Tmat[0], &Tmat[3]);
    A2D::VecAddCore<T,3>(-d, &Tmat[6], &Tmat[3]); // t1 - (t1 dot n) * n => t2 (temp store in t2)
    norm = sqrt(A2D::VecDotCore<T,3>(&Tmat[3], &Tmat[3]));
    A2D::VecAddCore<T,3>(1.0, &Tmat[3], &Tmat[0]);

    // compute t2 by cross product
    A2D::VecCrossCore<T>(&Tmat[6], &Tmat[0], &Tmat[3]); // nhat cross t1hat => t2hat

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

template <typename T, int vars_per_node, class Data, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrain(const T quad_pt[], const T refAxis[], const T xpts[],
                                             const T vars[], const T fn[], T et[]) {
    // TODO : do we actually need Ctn, Tn, XdinvTn, u0xn here?

    T etn[Basis::num_nodes];
    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        T pt[2];
        Basis::getNodePoint(inode, pt);

        // get shell transform and Xdn frame scope
        T Tmat[9], Xd[9];
        {
            // compute the computational coord gradients of Xpts for xi, eta
            T dXdxi[3], dXdeta[3];
            Basis::template interpFieldsGrad<3, 3>(pt, xpts, dXdxi, dXdeta);

            // assemble Xd frame
            Basis::assembleFrame(dXdxi, dXdeta, &fn[3 * inode], Xd);

            // compute the shell transform based on the ref axis in Data object
            ShellComputeTransform<T, Data>(refAxis, dXdxi, dXdeta, &fn[3 * inode], Tmat);
        }  // end of Xd and shell transform scope

        // assemble u0xn frame scope
        T u0xn[9];
        {
            // compute midplane disp field gradients
            T u0xi[3], u0eta[3];
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, u0xi, u0eta);

            A2D::Vec<T, 3> zero;
            Basis::assembleFrame(u0xi, u0eta, zero.get_data(), u0xn);
        }

        // compute rotation matrix at this node
        T C[9], tmp[9];
        Director::template computeRotationMat<vars_per_node, 1>(&vars[vars_per_node * inode], C);

        // compute Ct = T^T * C * T
        using MatOp = A2D::MatOp;
        A2D::MatMatMultCore3x3<T, MatOp::TRANSPOSE>(Tmat, C, tmp);
        A2D::MatMatMultCore3x3<T>(tmp, Tmat, C);

        // inverse Xd frame and Transformed product
        T XdinvTn[9];
        A2D::MatInvCore<T, 3>(Xd, tmp);
        A2D::MatMatMultCore3x3<T>(tmp, Tmat, XdinvTn);

        // Compute transformation u0x = T^T * u0xn * (Xdinv*T)
        A2D::MatMatMultCore3x3<T>(u0xn, XdinvTn, tmp);
        A2D::MatMatMultCore3x3<T, MatOp::TRANSPOSE>(Tmat, tmp, u0xn);

        // compute the drill strain
        etn[inode] = Director::evalDrillStrain(u0xn, C);

    }  // end of node for loop

    // now interpolate to single et value
    Basis::template interpFields<1, 1>(quad_pt, etn, et);

}  // end of method ShellComputeDrillStrain

template <typename T, int vars_per_node, class Data, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainV2(const T quad_pt[], const T refAxis[], const T xpts[],
                                             const T vars[], T et[]) {
    // TODO : do we actually need Ctn, Tn, XdinvTn, u0xn here?

    T etn[Basis::num_nodes];
    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        T pt[2];
        Basis::getNodePoint(inode, pt);

        // get shell transform and Xdn frame scope
        T Tmat[9], Xd[9], n0[3];
        {
            // compute the computational coord gradients of Xpts for xi, eta
            T dXdxi[3], dXdeta[3];
            Basis::template interpFieldsGrad<3, 3>(pt, xpts, dXdxi, dXdeta);

            // compute the shell node normals
            Basis::ShellComputeNodeNormal(pt, dXdxi, dXdeta, n0);

            // assemble Xd frame
            Basis::assembleFrame(dXdxi, dXdeta, n0, Xd);

            // compute the shell transform based on the ref axis in Data object
            ShellComputeTransform<T, Data>(refAxis, dXdxi, dXdeta, n0, Tmat);
        }  // end of Xd and shell transform scope

        // assemble u0xn frame scope
        T u0xn[9];
        {
            // compute midplane disp field gradients
            T u0xi[3], u0eta[3];
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, u0xi, u0eta);

            A2D::Vec<T, 3> zero;
            Basis::assembleFrame(u0xi, u0eta, zero.get_data(), u0xn);
        }

        // compute rotation matrix at this node
        T C[9], tmp[9];
        Director::template computeRotationMat<vars_per_node, 1>(&vars[vars_per_node * inode], C);

        // compute Ct = T^T * C * T
        using MatOp = A2D::MatOp;
        A2D::MatMatMultCore3x3<T, MatOp::TRANSPOSE>(Tmat, C, tmp);
        A2D::MatMatMultCore3x3<T>(tmp, Tmat, C);

        // inverse Xd frame and Transformed product
        T XdinvTn[9];
        A2D::MatInvCore<T, 3>(Xd, tmp);
        A2D::MatMatMultCore3x3<T>(tmp, Tmat, XdinvTn);

        // Compute transformation u0x = T^T * u0xn * (Xdinv*T)
        A2D::MatMatMultCore3x3<T>(u0xn, XdinvTn, tmp);
        A2D::MatMatMultCore3x3<T, MatOp::TRANSPOSE>(Tmat, tmp, u0xn);

        // compute the drill strain
        etn[inode] = Director::evalDrillStrain(u0xn, C);

    }  // end of node for loop

    // now interpolate to single et value
    Basis::template interpFields<1, 1>(quad_pt, etn, et);

}  // end of method ShellComputeDrillStrainV2

template <typename T, int vars_per_node, class Data, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainV3(const T quad_pt[], const T refAxis[], const T xpts[],
                                             const T vars[], T et[]) {
    // TODO : do we actually need Ctn, Tn, XdinvTn, u0xn here?

    T etn[Basis::num_nodes];
    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        T pt[2];
        Basis::getNodePoint(inode, pt);

        // get shell transform and Xdn frame scope
        T Tmat[9], Xd[9];
        {
            T n0[3];
            ShellComputeNodeNormalLight(pt, xpts, n0);

            // assemble Xd frame
            Basis::assembleFrameLight<3>(pt, xpts, n0, Xd);

            // compute the shell transform based on the ref axis in Data object
            ShellComputeTransformLight<T, Basis, Data>(refAxis, pt, xpts, n0, Tmat);
        }  // end of Xd and shell transform scope

        // assemble u0xn frame scope
        T u0xn[9];
        {
            Basis::assembleFrameLight<6>(pt, vars, &Xd[0], u0xn);
            for (int i = 0; i < 3; i++) u0xn[3*i+2] = 0.0; // last column is zero (just put Xd in there to save registers)
        }

        // compute rotation matrix at this node
        T C[9], tmp[9];
        Director::template computeRotationMat<vars_per_node, 1>(&vars[vars_per_node * inode], C);

        // compute Ct = T^T * C * T
        using MatOp = A2D::MatOp;
        A2D::MatMatMultCore3x3<T, MatOp::TRANSPOSE>(Tmat, C, tmp);
        A2D::MatMatMultCore3x3<T>(tmp, Tmat, C);

        // inverse Xd frame and Transformed product
        T XdinvTn[9];
        A2D::MatInvCore<T, 3>(Xd, tmp);
        A2D::MatMatMultCore3x3<T>(tmp, Tmat, XdinvTn);

        // Compute transformation u0x = T^T * u0xn * (Xdinv*T)
        A2D::MatMatMultCore3x3<T>(u0xn, XdinvTn, tmp);
        A2D::MatMatMultCore3x3<T, MatOp::TRANSPOSE>(Tmat, tmp, u0xn);

        // compute the drill strain
        etn[inode] = Director::evalDrillStrain(u0xn, C);

    }  // end of node for loop

    // now interpolate to single et value
    Basis::template interpFields<1, 1>(quad_pt, etn, et);

}  // end of method ShellComputeDrillStrainV3

template <typename T, int vars_per_node, class Data, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainFast(const T &xi, const T &eta, const T refAxis[], const T xpts[],
                                             const T vars[], T &et) {
    // instead of storing etn[4], we add to interpolated et on the fly..
    et = 0.0;
    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        // T pt[2];
        // Basis::getNodePoint(inode, pt);

        // // get shell transform and Xdn frame scope
        // T Tmat[9], Xd[9];
        // {
        //     T n0[3];
        //     ShellComputeNodeNormalLight(pt, xpts, n0);

        //     // assemble Xd frame
        //     Basis::assembleFrameLight<3>(pt, xpts, n0, Xd);

        //     // compute the shell transform based on the ref axis in Data object
        //     ShellComputeTransformLight<T, Basis, Data>(refAxis, pt, xpts, n0, Tmat);
        // }  // end of Xd and shell transform scope

        // // assemble u0xn frame scope
        // T u0xn[9];
        // {
        //     Basis::assembleFrameLight<6>(pt, vars, &Xd[0], u0xn);
        //     for (int i = 0; i < 3; i++) u0xn[3*i+2] = 0.0; // last column is zero (just put Xd in there to save registers)
        // }

        // // compute rotation matrix at this node
        // T C[9], tmp[9];
        // Director::template computeRotationMat<vars_per_node, 1>(&vars[vars_per_node * inode], C);

        // // compute Ct = T^T * C * T
        // using MatOp = A2D::MatOp;
        // A2D::MatMatMultCore3x3<T, MatOp::TRANSPOSE>(Tmat, C, tmp);
        // A2D::MatMatMultCore3x3<T>(tmp, Tmat, C);

        // // inverse Xd frame and Transformed product
        // T XdinvTn[9];
        // A2D::MatInvCore<T, 3>(Xd, tmp);
        // A2D::MatMatMultCore3x3<T>(tmp, Tmat, XdinvTn);

        // // Compute transformation u0x = T^T * u0xn * (Xdinv*T)
        // A2D::MatMatMultCore3x3<T>(u0xn, XdinvTn, tmp);
        // A2D::MatMatMultCore3x3<T, MatOp::TRANSPOSE>(Tmat, tmp, u0xn);

        // compute the drill strain
        // T etn = Director::evalDrillStrain(u0xn, C);

        // add to interp of et_f

    }  // end of node for loop

}  // end of method ShellComputeDrillStrainFast


template <typename T, int vars_per_node, class Data, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainHfwd(const T quad_pt[], const T refAxis[],
                                                 const T xpts[], const T pvars[], const T fn[],
                                                 T et_dot[]) {
    // since it's linear just equiv to forward analysis with pvars input
    ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(quad_pt, refAxis, xpts, pvars,
                                                                     fn, et_dot);

}  // end of method ShellComputeDrillStrainHfwd

template <typename T, int vars_per_node, class Data, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainSens(const T quad_pt[], const T refAxis[],
                                                 const T xpts[], const T vars[], const T fn[],
                                                 const T et_bar[], T res[]) {
    // TODO : do we actually need Ctn, Tn, XdinvTn, u0xn here?

    // first interpolate back to nodal level
    A2D::Vec<T, Basis::num_nodes> etn_bar;
    Basis::template interpFieldsTranspose<1, 1>(quad_pt, et_bar, etn_bar.get_data());

    constexpr A2D::MatOp NORM = A2D::MatOp::NORMAL;
    constexpr A2D::MatOp TRANS = A2D::MatOp::TRANSPOSE;

    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        T pt[2];
        Basis::getNodePoint(inode, pt);

        // get shell transform and Xdn frame scope
        T Tmat[9], Xd[9];
        {
            // compute the computational coord gradients of Xpts for xi, eta
            T dXdxi[3], dXdeta[3];
            Basis::template interpFieldsGrad<3, 3>(pt, xpts, dXdxi, dXdeta);

            // assemble Xd frame
            Basis::assembleFrame(dXdxi, dXdeta, &fn[3 * inode], Xd);

            // compute the shell transform based on the ref axis in Data object
            ShellComputeTransform<T, Data>(refAxis, dXdxi, dXdeta, &fn[3 * inode], Tmat);
        }  // end of Xd and shell transform scope

        T u0xn_bar[9], C_bar[9];  // really u0x_bar at this point (but don't want two vars for it)
        Director::evalDrillStrainSens(etn_bar[inode], u0xn_bar, C_bar);

        T tmp[9];
        {
            // C_bar = T * C2_bar * T^t (in reverse)
            A2D::MatMatMultCore3x3<T>(Tmat, C_bar, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, C_bar);

            // reverse C(vars) to C_bar => res
            Director::template computeRotationMatSens<vars_per_node, 1>(
                C_bar, &res[vars_per_node * inode]);
        }

        // backprop u0x_bar to u0xn_bar^T
        {
            T XdinvT[9];
            A2D::MatInvCore<T, 3>(Xd, tmp);  // Xdinv
            A2D::MatMatMultCore3x3<T>(tmp, Tmat, XdinvT);

            // u0xn_bar = T * u0x_bar * XdinvT^t
            // transpose version for convenience of cols avail in rows
            // u0xn_bar^t = XdinvT * u0x_bar^t * T^t (u0xn_bar now holds transpose u0xn_bar^t)
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvT, u0xn_bar, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, u0xn_bar);

            // reverse the interpolations u0xn_bar to res
            // because we have u0xn_bar^T stored, each row is u0xi_bar, u0eta_bar
            Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, &u0xn_bar[0],
                                                                        &u0xn_bar[3], res);
        }

    }  // end of node for loop
}  // end of method ShellComputeDrillStrainSens

template <typename T, int vars_per_node, class Data, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainSensV2(const T quad_pt[], const T refAxis[],
                                                 const T xpts[], const T vars[],
                                                 const T et_bar[], T res[]) {
    // TODO : do we actually need Ctn, Tn, XdinvTn, u0xn here?

    // first interpolate back to nodal level
    A2D::Vec<T, Basis::num_nodes> etn_bar;
    Basis::template interpFieldsTranspose<1, 1>(quad_pt, et_bar, etn_bar.get_data());

    constexpr A2D::MatOp NORM = A2D::MatOp::NORMAL;
    constexpr A2D::MatOp TRANS = A2D::MatOp::TRANSPOSE;

    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        T pt[2];
        Basis::getNodePoint(inode, pt);

        // get shell transform and Xdn frame scope
        T Tmat[9], Xd[9], n0[3];
        {
            // compute the computational coord gradients of Xpts for xi, eta
            T dXdxi[3], dXdeta[3];
            Basis::template interpFieldsGrad<3, 3>(pt, xpts, dXdxi, dXdeta);

            // compute the shell node normals
            Basis::ShellComputeNodeNormal(pt, dXdxi, dXdeta, n0);

            // assemble Xd frame
            Basis::assembleFrame(dXdxi, dXdeta, n0, Xd);

            // compute the shell transform based on the ref axis in Data object
            ShellComputeTransform<T, Data>(refAxis, dXdxi, dXdeta, n0, Tmat);
        }  // end of Xd and shell transform scope

        T u0xn_bar[9], C_bar[9];  // really u0x_bar at this point (but don't want two vars for it)
        Director::evalDrillStrainSens(etn_bar[inode], u0xn_bar, C_bar);

        T tmp[9];
        {
            // C_bar = T * C2_bar * T^t (in reverse)
            A2D::MatMatMultCore3x3<T>(Tmat, C_bar, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, C_bar);

            // reverse C(vars) to C_bar => res
            Director::template computeRotationMatSens<vars_per_node, 1>(
                C_bar, &res[vars_per_node * inode]);
        }

        // backprop u0x_bar to u0xn_bar^T
        {
            T XdinvT[9];
            A2D::MatInvCore<T, 3>(Xd, tmp);  // Xdinv
            A2D::MatMatMultCore3x3<T>(tmp, Tmat, XdinvT);

            // u0xn_bar = T * u0x_bar * XdinvT^t
            // transpose version for convenience of cols avail in rows
            // u0xn_bar^t = XdinvT * u0x_bar^t * T^t (u0xn_bar now holds transpose u0xn_bar^t)
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvT, u0xn_bar, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, u0xn_bar);

            // reverse the interpolations u0xn_bar to res
            // because we have u0xn_bar^T stored, each row is u0xi_bar, u0eta_bar
            Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, &u0xn_bar[0],
                                                                        &u0xn_bar[3], res);
        }

    }  // end of node for loop
}  // end of method ShellComputeDrillStrainSensV2

template <typename T, int vars_per_node, class Data, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainSensV3(const T quad_pt[], const T refAxis[],
                                                 const T xpts[], const T vars[],
                                                 const T et_bar[], T res[]) {
    // TODO : do we actually need Ctn, Tn, XdinvTn, u0xn here?

    // first interpolate back to nodal level
    A2D::Vec<T, Basis::num_nodes> etn_bar;
    Basis::template interpFieldsTransposeLight<1, 1>(quad_pt, et_bar, etn_bar.get_data());

    constexpr A2D::MatOp NORM = A2D::MatOp::NORMAL;
    constexpr A2D::MatOp TRANS = A2D::MatOp::TRANSPOSE;

    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        T pt[2];
        Basis::getNodePoint(inode, pt);

        // get shell transform and Xdn frame scope
        T Tmat[9], Xd[9];
        {
            T n0[3];
            ShellComputeNodeNormalLight(pt, xpts, n0);

            // assemble Xd frame
            Basis::assembleFrameLight<3>(pt, xpts, n0, Xd);

            // compute the shell transform based on the ref axis in Data object
            ShellComputeTransformLight<T, Basis, Data>(refAxis, pt, xpts, n0, Tmat);
        }  // end of Xd and shell transform scope

        T u0xn_bar[9], C_bar[9];  // really u0x_bar at this point (but don't want two vars for it)
        Director::evalDrillStrainSens(etn_bar[inode], u0xn_bar, C_bar);

        T tmp[9];
        {
            // C_bar = T * C2_bar * T^t (in reverse)
            A2D::MatMatMultCore3x3<T>(Tmat, C_bar, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, C_bar);

            // reverse C(vars) to C_bar => res
            Director::template computeRotationMatSens<vars_per_node, 1>(
                C_bar, &res[vars_per_node * inode]);
        }

        // backprop u0x_bar to u0xn_bar^T
        {
            T XdinvT[9];
            A2D::MatInvCore<T, 3>(Xd, tmp);  // Xdinv
            A2D::MatMatMultCore3x3<T>(tmp, Tmat, XdinvT);

            // u0xn_bar = T * u0x_bar * XdinvT^t
            // transpose version for convenience of cols avail in rows
            // u0xn_bar^t = XdinvT * u0x_bar^t * T^t (u0xn_bar now holds transpose u0xn_bar^t)
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvT, u0xn_bar, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, u0xn_bar);

            // reverse the interpolations u0xn_bar to res
            // because we have u0xn_bar^T stored, each row is u0xi_bar, u0eta_bar
            Basis::template interpFieldsGradTransposeLight<vars_per_node, 3>(pt, &u0xn_bar[0],
                                                                        &u0xn_bar[3], res);
        }

    }  // end of node for loop
}  // end of method ShellComputeDrillStrainSensV3

template <typename T, int vars_per_node, class Data, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainHrev(const T quad_pt[], const T refAxis[],
                                                 const T xpts[], const T vars[], const T fn[],
                                                 const T et_hat[], T matCol[]) {
    // since this is a purely linear function, same backprop rule as 1st derivs
    ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(quad_pt, refAxis, xpts,
                                                                         vars, fn, et_hat, matCol);

}  // end of method ShellComputeDrillStrainHrev

template <typename T, class Physics, class Basis, bool is_nonlinear>
__HOST_DEVICE__ static void computeTyingStrain(const T Xpts[], const T fn[], const T vars[],
                                               const T d[], T ety[]) {
    // using unrolled loop here for efficiency (if statements and for loops not
    // great for device)
    int32_t offset, num_tying;
    static constexpr int vars_per_node = Physics::vars_per_node;

    // get g11 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(0);
    num_tying = Basis::num_tying_points(0);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<0>(itying, pt);

        // Interpolate the field value
        T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

        // store g11 strain
        ety[offset + itying] = A2D::VecDotCore<T, 3>(Uxi, Xxi);

        // nonlinear g11 strain term
        if constexpr (is_nonlinear) {
            ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Uxi, Uxi);
        }

    }  // end of itying for loop for g11

    // get g22 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(1);
    num_tying = Basis::num_tying_points(1);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<1>(itying, pt);

        // Interpolate the field value
        T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

        // store g22 strain
        ety[offset + itying] = A2D::VecDotCore<T, 3>(Ueta, Xeta);

        // nonlinear g22 strain term
        if constexpr (is_nonlinear) {
            ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Ueta, Ueta);
        }

    }  // end of itying for loop for g22

    // get g12 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(2);
    num_tying = Basis::num_tying_points(2);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<2>(itying, pt);

        // Interpolate the field value
        T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

        // store g12 strain
        ety[offset + itying] =
            0.5 * (A2D::VecDotCore<T, 3>(Uxi, Xeta) + A2D::VecDotCore<T, 3>(Ueta, Xxi));

        // nonlinear g12 strain term
        if constexpr (is_nonlinear) {
            ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Uxi, Ueta);
        }
    }  // end of itying for loop for g12

    // get g23 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(3);
    num_tying = Basis::num_tying_points(3);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<3>(itying, pt);

        // Interpolate the field value
        T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

        T d0[3], n0[3];
        Basis::template interpFields<3, 3>(pt, d, d0);
        Basis::template interpFields<3, 3>(pt, fn, n0);

        // store g23 strain
        ety[offset + itying] =
            0.5 * (A2D::VecDotCore<T, 3>(Xeta, d0) + A2D::VecDotCore<T, 3>(n0, Ueta));

        // nonlinear g23 strain term
        if constexpr (is_nonlinear) {
            ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(d0, Ueta);
        }
    }  // end of itying for loop for g23

    // get g13 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(4);
    num_tying = Basis::num_tying_points(4);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<4>(itying, pt);

        // Interpolate the field value
        T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

        T d0[3], n0[3];
        Basis::template interpFields<3, 3>(pt, d, d0);
        Basis::template interpFields<3, 3>(pt, fn, n0);

        // store g13 strain
        ety[offset + itying] =
            0.5 * (A2D::VecDotCore<T, 3>(Xxi, d0) + A2D::VecDotCore<T, 3>(n0, Uxi));

        // nonlinear g13 strain term
        if constexpr (is_nonlinear) {
            ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(d0, Uxi);
        }
    }  // end of itying for loop for g13

}  // end of computeTyingStrain

template <typename T, class Physics, class Basis, class Director>
__HOST_DEVICE__ static void computeTyingStrainLight(const T xpts[], const T vars[],
                                               T ety[]) {
    // using unrolled loop here for efficiency (if statements and for loops not
    // great for device)
    int32_t offset, num_tying;
    static constexpr bool is_nonlinear = Physics::is_nonlinear;
    static constexpr int vars_per_node = Physics::vars_per_node;

    // get g11 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(0);
    num_tying = Basis::num_tying_points(0);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<0>(itying, pt);

        // g11 strain = X,xi * U0,xi (midplane disps)
        ety[offset + itying] = Basis::interpFieldsGradDotLight<XI, XI, 3, vars_per_node, 3>(pt, xpts, vars);

        // nonlinear g11 strain term = 0.5 * U0,xi * U0,xi
        if constexpr (is_nonlinear) {
            ety[offset + itying] += 0.5 * Basis::interpFieldsGradDotLight<XI, XI, vars_per_node, vars_per_node, 3>(pt, vars, vars);
        }

    }  // end of itying for loop for g11

    // get g22 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(1);
    num_tying = Basis::num_tying_points(1);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<1>(itying, pt);

        // store g22 strain = X,eta * U0,eta
        ety[offset + itying] = Basis::interpFieldsGradDotLight<ETA, ETA, 3, vars_per_node, 3>(pt, xpts, vars);

        // nonlinear g22 strain term += 0.5 * U0,eta * U0,eta
        if constexpr (is_nonlinear) {
            ety[offset + itying] += 0.5 * Basis::interpFieldsGradDotLight<ETA, ETA, vars_per_node, vars_per_node, 3>(pt, vars, vars);
        }

    }  // end of itying for loop for g22

    // get g12 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(2);
    num_tying = Basis::num_tying_points(2);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<2>(itying, pt);

        // store g12 strain = 0.5 * (X,eta * U0,xi + X,xi * U0,eta)
        ety[offset + itying] = 0.5 * Basis::interpFieldsGradDotLight<ETA, XI, 3, vars_per_node, 3>(pt, xpts, vars);
        ety[offset + itying] += 0.5 * Basis::interpFieldsGradDotLight<XI, ETA, 3, vars_per_node, 3>(pt, xpts, vars);

        // nonlinear g12 strain term += 0.5 * U0,xi * U0,eta
        if constexpr (is_nonlinear) {
            ety[offset + itying] += 0.5 * Basis::interpFieldsGradDotLight<XI, ETA, vars_per_node, vars_per_node, 3>(pt, vars, vars);
        }
    }  // end of itying for loop for g12

    // get g23 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(3);
    num_tying = Basis::num_tying_points(3);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<3>(itying, pt);

        // store g23 strain = 0.5 * (X,eta * d0 + U,eta * n0)
        T d0[3]; // X,eta dot d0 term
        Director::template interpDirectorLight<Basis, vars_per_node, Basis::num_nodes>(pt, xpts, vars, d0);
        ety[offset + itying] = 0.5 * Basis::interpFieldsGradRightDotLight<ETA, 3, 3>(pt, xpts, d0);
        {
            T n0[3]; // U0,eta dot d0 term
            Basis::interpNodeNormalLight(pt, xpts, n0);
            ety[offset + itying] += 0.5 * Basis::interpFieldsGradRightDotLight<ETA, vars_per_node, 3>(pt, vars, n0);
        }

        // nonlinear g23 strain term += 0.5 * U,eta dot d0
        if constexpr (is_nonlinear) {
            ety[offset + itying] += 0.5 * Basis::interpFieldsGradRightDotLight<ETA, vars_per_node, 3>(pt, vars, d0);
        }
    }  // end of itying for loop for g23

    // get g13 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(4);
    num_tying = Basis::num_tying_points(4);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<4>(itying, pt);

        // store g13 strain = 0.5 * (X,xi * d0 + U,xi * n0)
        T d0[3]; // X,eta dot d0 term
        Director::template interpDirectorLight<Basis, vars_per_node, Basis::num_nodes>(pt, xpts, vars, d0);
        ety[offset + itying] = 0.5 * Basis::interpFieldsGradRightDotLight<XI, 3, 3>(pt, xpts, d0);
        {
            T n0[3]; // U0,eta dot d0 term
            Basis::interpNodeNormalLight(pt, xpts, n0);
            ety[offset + itying] += 0.5 * Basis::interpFieldsGradRightDotLight<XI, vars_per_node, 3>(pt, vars, n0);
        }

        // nonlinear g13 strain term += 0.5 * U,xi dot d0
        if constexpr (is_nonlinear) {
            ety[offset + itying] += 0.5 * Basis::interpFieldsGradRightDotLight<XI, vars_per_node, 3>(pt, vars, d0);
        }
    }  // end of itying for loop for g13

}  // end of computeTyingStrain

template <typename T, class Physics, class Basis>
__HOST_DEVICE__ static void computeTyingStrainHfwd(const T Xpts[], const T fn[], const T vars[],
                                                   const T d[], const T p_vars[], const T p_d[],
                                                   T p_ety[]) {
    // linear part
    computeTyingStrain<T, Physics, Basis, false>(Xpts, fn, p_vars, p_d, p_ety);

    // using unrolled loop here for efficiency (if statements and for loops not
    // great for device)
    int32_t offset, num_tying;
    static constexpr bool is_nonlinear = Physics::is_nonlinear;
    static constexpr int vars_per_node = Physics::vars_per_node;

    if constexpr (!is_nonlinear) {
        return;  // exit early for linear part
    }
    // return;  // temporary debug check

    // remaining nonlinear extra terms of Hfwd

    // get g11 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(0);
    num_tying = Basis::num_tying_points(0);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<0>(itying, pt);

        // Interpolate the field value
        T Uxi[3], p_Uxi[3], zero[3];
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, zero);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, p_Uxi, zero);
        p_ety[offset + itying] += A2D::VecDotCore<T, 3>(Uxi, p_Uxi);

    }  // end of itying for loop for g11

    // get g22 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(1);
    num_tying = Basis::num_tying_points(1);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<1>(itying, pt);

        // Interpolate the field value
        T Ueta[3], p_Ueta[3], zero[3];
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, zero, Ueta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, zero, p_Ueta);
        p_ety[offset + itying] += A2D::VecDotCore<T, 3>(Ueta, p_Ueta);

    }  // end of itying for loop for g22

    // get g12 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(2);
    num_tying = Basis::num_tying_points(2);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<2>(itying, pt);

        // Interpolate the field value
        T Uxi[3], Ueta[3], p_Uxi[3], p_Ueta[3];
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, p_Uxi, p_Ueta);
        p_ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Uxi, p_Ueta);
        p_ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Ueta, p_Uxi);
    }  // end of itying for loop for g12

    // get g23 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(3);
    num_tying = Basis::num_tying_points(3);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<3>(itying, pt);

        // Interpolate the field value
        T Ueta[3], p_Ueta[3], zero[3], d0[3], p_d0[3];
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, zero, Ueta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, zero, p_Ueta);
        Basis::template interpFields<3, 3>(pt, d, d0);
        Basis::template interpFields<3, 3>(pt, p_d, p_d0);
        p_ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(d0, p_Ueta);
        p_ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(p_d0, Ueta);
    }  // end of itying for loop for g23

    // get g13 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(4);
    num_tying = Basis::num_tying_points(4);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<4>(itying, pt);

        // Interpolate the field value
        T Uxi[3], p_Uxi[3], zero[3], d0[3], p_d0[3];
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, zero);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, p_Uxi, zero);
        Basis::template interpFields<3, 3>(pt, d, d0);
        Basis::template interpFields<3, 3>(pt, p_d, p_d0);
        p_ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(d0, p_Uxi);
        p_ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(p_d0, Uxi);
    }  // end of itying for loop for g13
}

template <typename T, class Physics, class Basis, class Director>
__HOST_DEVICE__ static void computeTyingStrainSens(const T Xpts[], const T fn[], const T vars[],
                                                   const T d[], const T ety_bar[], T res[],
                                                   T d_bar[]) {
    // using unrolled loop here for efficiency (if statements and for loops not
    // great for device)
    int32_t offset, num_tying;
    static constexpr bool is_nonlinear = Physics::is_nonlinear;
    static constexpr int vars_per_node = Physics::vars_per_node;

    // get g11 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(0);
    num_tying = Basis::num_tying_points(0);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<0>(itying, pt);
        //   ety[offset + itying] = A2D::VecDotCore<T, 3>(Uxi, Xxi);

        T Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);

        A2D::Vec<T, 3> Uxi_bar, zero;
        A2D::VecAddCore<T, 3>(ety_bar[offset + itying], Xxi, Uxi_bar.get_data());

        // nonlinear g11 strain term backprop
        if constexpr (is_nonlinear) {
            T Uxi[3], Ueta[3];
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);
            // recall forward analysis nonlinear g11 term:
            // ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Uxi, Uxi);
            A2D::VecAddCore<T, 3>(ety_bar[offset + itying], Uxi, Uxi_bar.get_data());
        }

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_bar.get_data(),
                                                                    zero.get_data(), res);

    }  // end of itying for loop for g11

    // get g22 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(1);
    num_tying = Basis::num_tying_points(1);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<1>(itying, pt);
        //   ety[offset + itying] = A2D::VecDotCore<T, 3>(Ueta, Xeta);

        T Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);

        A2D::Vec<T, 3> Ueta_bar, zero;
        A2D::VecAddCore<T, 3>(ety_bar[offset + itying], Xeta, Ueta_bar.get_data());

        // nonlinear g22 strain term backprop
        if constexpr (is_nonlinear) {
            T Uxi[3], Ueta[3];
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);
            // recall forward analysis nonlinear g22 term:
            // ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Ueta, Ueta);
            A2D::VecAddCore<T, 3>(ety_bar[offset + itying], Ueta, Ueta_bar.get_data());
        }

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, zero.get_data(),
                                                                    Ueta_bar.get_data(), res);

    }  // end of itying for loop for g22

    // get g12 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(2);
    num_tying = Basis::num_tying_points(2);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<2>(itying, pt);
        //   ety[offset + itying] = 0.5 * (A2D::VecDotCore<T, 3>(Uxi, Xeta) +
        //                                 A2D::VecDotCore<T, 3>(Ueta, Xxi));

        T Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);

        A2D::Vec<T, 3> Uxi_bar, Ueta_bar;
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Xxi, Ueta_bar.get_data());
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Xeta, Uxi_bar.get_data());

        // nonlinear g12 strain term backprop
        if constexpr (is_nonlinear) {
            T Uxi[3], Ueta[3];
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);
            // recall forward analysis nonlinear g12 term:
            // ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Uxi, Ueta);
            A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Uxi, Ueta_bar.get_data());
            A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Ueta, Uxi_bar.get_data());
        }

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_bar.get_data(),
                                                                    Ueta_bar.get_data(), res);
    }  // end of itying for loop for g12

    // get g23 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(3);
    num_tying = Basis::num_tying_points(3);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<3>(itying, pt);
        //   ety[offset + itying] = 0.5 * (A2D::VecDotCore<T, 3>(Xeta, d0) +
        //                                 A2D::VecDotCore<T, 3>(n0, Ueta));

        T Xxi[3], Xeta[3], n0[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
        Basis::template interpFields<3, 3>(pt, fn, n0);

        A2D::Vec<T, 3> zero, d0_bar, Ueta_bar;
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Xeta, d0_bar.get_data());
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], n0, Ueta_bar.get_data());

        // nonlinear g23 strain term backprop
        if constexpr (is_nonlinear) {
            T d0[3];
            Basis::template interpFields<3, 3>(pt, d, d0);
            T Uxi[3], Ueta[3];
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);
            // recall forward analysis nonlinear g23 term:
            // ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(d0, Ueta);
            A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Ueta, d0_bar.get_data());
            A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], d0, Ueta_bar.get_data());
        }

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, zero.get_data(),
                                                                    Ueta_bar.get_data(), res);
        Basis::template interpFieldsTranspose<3, 3>(pt, d0_bar.get_data(), d_bar);
    }  // end of itying for loop for g23

    // get g13 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(4);
    num_tying = Basis::num_tying_points(4);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<4>(itying, pt);
        //   ety[offset + itying] = 0.5 * (A2D::VecDotCore<T, 3>(Xxi, d0) +
        //                                 A2D::VecDotCore<T, 3>(n0, Uxi));

        T Xxi[3], Xeta[3], n0[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
        Basis::template interpFields<3, 3>(pt, fn, n0);

        A2D::Vec<T, 3> zero, Uxi_bar, d0_bar;
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Xxi, d0_bar.get_data());
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], n0, Uxi_bar.get_data());

        // nonlinear g23 strain term backprop
        if constexpr (is_nonlinear) {
            T d0[3];
            Basis::template interpFields<3, 3>(pt, d, d0);
            T Uxi[3], Ueta[3];
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);
            // recall forward analysis nonlinear g22 term:
            // ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(d0, Uxi);
            A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Uxi, d0_bar.get_data());
            A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], d0, Uxi_bar.get_data());
        }

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_bar.get_data(),
                                                                    zero.get_data(), res);
        Basis::template interpFieldsTranspose<3, 3>(pt, d0_bar.get_data(), d_bar);

    }  // end of itying for loop for g13

}  // end of computeTyingStrainSens

template <typename T, class Physics, class Basis, class Director>
__HOST_DEVICE__ static void computeTyingStrainSensLight(const T xpts[], const T vars[],
                                                   const T ety_bar[], T res[]) {
    // using unrolled loop here for efficiency (if statements and for loops not
    // great for device)
    int32_t offset, num_tying;
    static constexpr bool is_nonlinear = Physics::is_nonlinear;
    static constexpr int vars_per_node = Physics::vars_per_node;

    // get g11 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(0);
    num_tying = Basis::num_tying_points(0);

    // return; // 32 registers per thread 

#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<0>(itying, pt);
        
	// return; // still 32 registers per thread	

	// backprop g11 = X,xi dot U0,xi
        Basis::template addInterpFieldsGradDotSensLight<XI, XI, 3, vars_per_node, 3>(ety_bar[offset + itying], pt, xpts, res);

        return; // 136 registers per thread?

        if constexpr (is_nonlinear) {
            // backprop g11 nl term 1/2 * U0,xi dot U0,xi
            Basis::template addInterpFieldsGradDotSensLight<XI, XI, vars_per_node, vars_per_node, 3>(ety_bar[offset + itying], pt, vars, res);
        }
    }  // end of itying for loop for g11

    // get g22 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(1);
    num_tying = Basis::num_tying_points(1);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<1>(itying, pt);
        // backprop g22 = X,eta dot U0,eta
        Basis::template addInterpFieldsGradDotSensLight<ETA, ETA, 3, vars_per_node, 3>(ety_bar[offset + itying], pt, xpts, res);

        if constexpr (is_nonlinear) {
            // backprop g22 nl term 1/2 * U0,eta dot U0,eta
            Basis::template addInterpFieldsGradDotSensLight<ETA, ETA, vars_per_node, vars_per_node, 3>(ety_bar[offset + itying], pt, vars, res);
        }
    }  // end of itying for loop for g22

    // get g12 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(2);
    num_tying = Basis::num_tying_points(2);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<2>(itying, pt);

        // backprop g12 = 0.5 * (X,eta dot U0,xi + X,xi dot U0,eta)
        Basis::template addInterpFieldsGradDotSensLight<XI, ETA, 3, vars_per_node, 3>(0.5 * ety_bar[offset + itying], pt, xpts, res);
        Basis::template addInterpFieldsGradDotSensLight<ETA, XI, 3, vars_per_node, 3>(0.5 * ety_bar[offset + itying], pt, xpts, res);

        if constexpr (is_nonlinear) {
            // backprop g12 nl term 1/2 * U0,xi dot U0,eta
            Basis::template addInterpFieldsGradDotSensLight<XI, ETA, vars_per_node, vars_per_node, 3>(ety_bar[offset + itying], pt, vars, res);
            Basis::template addInterpFieldsGradDotSensLight<ETA, XI, vars_per_node, vars_per_node, 3>(ety_bar[offset + itying], pt, vars, res);
        }
    }  // end of itying for loop for g12

    // get g23 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(3);
    num_tying = Basis::num_tying_points(3);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<3>(itying, pt);

        // g23 = 0.5 * (X,eta dot d0 + n0 dot U,eta)
        A2D::Vec<T,3> d0_bar;
        Basis::template interpFieldsGradRightDotLight_RightSens<ETA, 3, 3>(0.5 * ety_bar[offset + itying], pt, xpts, d0_bar.get_data());
        {
            T n0[3]; // U0,eta dot d0 term
            Basis::template interpNodeNormalLight(pt, xpts, n0);
            Basis::template interpFieldsGradRightDotLight_LeftSens<ETA, vars_per_node, 3>(0.5 * ety_bar[offset + itying], pt, res, n0);
        }

        // nonlinear g23 strain term += 0.5 * U,eta dot d0
        if constexpr (is_nonlinear) {
            T d0[3];
            Director::template interpDirectorLight<Basis, vars_per_node, Basis::num_nodes>(pt, xpts, vars, d0);

            Basis::template interpFieldsGradRightDotLight_LeftSens<ETA, vars_per_node, 3>(0.5 * ety_bar[offset + itying], pt, res, d0);
            Basis::template interpFieldsGradRightDotLight_RightSens<ETA, vars_per_node, 3>(0.5 * ety_bar[offset + itying], pt, vars, d0_bar.get_data());
        }
        Director::template interpDirectorLightSens<Basis, vars_per_node, Basis::num_nodes>(1.0, pt, xpts, d0_bar.get_data(), res);
    }  // end of itying for loop for g23

    // get g13 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(4);
    num_tying = Basis::num_tying_points(4);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<4>(itying, pt);
        
        // g23 = 0.5 * (X,xi dot d0 + n0 dot U,xi)
        A2D::Vec<T,3> d0_bar;
        Basis::template interpFieldsGradRightDotLight_RightSens<XI, 3, 3>(0.5 * ety_bar[offset + itying], pt, xpts, d0_bar.get_data());
        {
            T n0[3]; // U0,eta dot d0 term
            Basis::template interpNodeNormalLight(pt, xpts, n0);
            Basis::template interpFieldsGradRightDotLight_LeftSens<XI, vars_per_node, 3>(0.5 * ety_bar[offset + itying], pt, res, n0);
        }

        // nonlinear g23 strain term += 0.5 * U,eta dot d0
        if constexpr (is_nonlinear) {
            T d0[3];
            Director::template interpDirectorLight<Basis, vars_per_node, Basis::num_nodes>(pt, xpts, vars, d0);

            Basis::template interpFieldsGradRightDotLight_LeftSens<XI, vars_per_node, 3>(0.5 * ety_bar[offset + itying], pt, res, d0);
            Basis::template interpFieldsGradRightDotLight_RightSens<XI, vars_per_node, 3>(0.5 * ety_bar[offset + itying], pt, vars, d0_bar.get_data());
        }
        Director::template interpDirectorLightSens<Basis, vars_per_node, Basis::num_nodes>(1.0, pt, xpts, d0_bar.get_data(), res);
    }  // end of itying for loop for g13

}  // end of computeTyingStrainSens

template <typename T, class Physics, class Basis>
__HOST_DEVICE__ static void computeTyingStrainHrev(const T Xpts[], const T fn[], const T vars[],
                                                   const T d[], const T p_vars[], const T p_d[],
                                                   const T ety_bar[], const T h_ety[], T matCol[],
                                                   T h_d[]) {
    // 2nd order backprop terms, linear part
    computeTyingStrainSens<T, Physics, Basis>(Xpts, fn, vars, d, h_ety, matCol, h_d);

    static constexpr bool is_nonlinear = Physics::is_nonlinear;

    if constexpr (!is_nonlinear) {
        return;
    }
    // return;  // temp debug

    // remaining part is nonlinear mixed term
    // ybar_i * d^2y_i/dxk/dxl * xdot_l mixed term with forward inputs here
    // only valid for nonlinear case
    int32_t offset, num_tying;
    static constexpr int vars_per_node = Physics::vars_per_node;

    // get g11 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(0);
    num_tying = Basis::num_tying_points(0);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<0>(itying, pt);
        //   ety[offset + itying] NL = 1/2 * Uxi dot Uxi;
        // g11_bar * d^2g11/dUxi/dUxi * p_Uxi term

        T p_Uxi[3];
        A2D::Vec<T, 3> Uxi_hat, zero;
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, p_Uxi, zero.get_data());
        // had 1/2 * ety_bar before but now I'm trying 1 since looks like sens above
        A2D::VecAddCore<T, 3>(ety_bar[offset + itying], p_Uxi, Uxi_hat.get_data());

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_hat.get_data(),
                                                                    zero.get_data(), matCol);

    }  // end of itying for loop for g11

    // return;  // temp debug

    // get g22 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(1);
    num_tying = Basis::num_tying_points(1);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<1>(itying, pt);
        //   ety[offset + itying] NL = 1/2 * Ueta dot Ueta;
        // g22_bar * d^2g22/dUeta/dUeta * p_Ueta term

        T p_Ueta[3];
        A2D::Vec<T, 3> Ueta_hat, zero;
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, zero.get_data(), p_Ueta);
        // had 1/2 * ety_bar before but now I'm trying 1 since looks like sens above
        A2D::VecAddCore<T, 3>(ety_bar[offset + itying], p_Ueta, Ueta_hat.get_data());

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, zero.get_data(),
                                                                    Ueta_hat.get_data(), matCol);

    }  // end of itying for loop for g22

    // get g12 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(2);
    num_tying = Basis::num_tying_points(2);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<2>(itying, pt);
        // ety[offset + itying] NL = 1/2 * Uxi dot Ueta
        // g12_bar * d^2g22/dUxi/dUeta * p_Uxi + again but swap Ueta,Uxi (term)

        T p_Uxi[3], p_Ueta[3];
        A2D::Vec<T, 3> Uxi_hat, Ueta_hat;
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, p_Uxi, p_Ueta);
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], p_Uxi, Ueta_hat.get_data());
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], p_Ueta, Uxi_hat.get_data());

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_hat.get_data(),
                                                                    Ueta_hat.get_data(), matCol);
    }  // end of itying for loop for g12

    // get g23 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(3);
    num_tying = Basis::num_tying_points(3);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<3>(itying, pt);
        // ety[offset + itying] NL = 1/2 * d0 dot Ueta
        // g23_bar * d^2g22/dd0/dUeta * p_Ueta + again but swap Ueta,d0 (term)

        T p_Ueta[3];
        A2D::Vec<T, 3> d0_hat, Ueta_hat, zero;
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, zero.get_data(), p_Ueta);
        T p_d0[3];
        Basis::template interpFields<3, 3>(pt, p_d, p_d0);

        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], p_d0, Ueta_hat.get_data());
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], p_Ueta, d0_hat.get_data());

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, zero.get_data(),
                                                                    Ueta_hat.get_data(), matCol);
        Basis::template interpFieldsTranspose<3, 3>(pt, d0_hat.get_data(), h_d);
    }  // end of itying for loop for g23

    // get g13 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(4);
    num_tying = Basis::num_tying_points(4);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<4>(itying, pt);
        // ety[offset + itying] NL = 1/2 * d0 dot Uxi
        // g13_bar * d^2g22/dd0/dUxi * p_Uxi + again but swap Uxi,d0 (term)

        T p_Uxi[3];
        A2D::Vec<T, 3> d0_hat, Uxi_hat, zero;
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, p_Uxi, zero.get_data());
        T p_d0[3];
        Basis::template interpFields<3, 3>(pt, p_d, p_d0);

        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], p_d0, Uxi_hat.get_data());
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], p_Uxi, d0_hat.get_data());

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_hat.get_data(),
                                                                    zero.get_data(), matCol);
        Basis::template interpFieldsTranspose<3, 3>(pt, d0_hat.get_data(), h_d);

    }  // end of itying for loop for g13
}
