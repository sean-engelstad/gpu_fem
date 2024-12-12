#pragma once
#include "../cuda_utils.h"

template <typename T, class Data>
__HOST_DEVICE__ void ShellComputeTransform(const T refAxis[], const T dXdxi[], const T dXdeta[], const T n0[], T Tmat[]) {
    
    A2D::Vec<T,3> n(n0), nhat;
    A2D::VecNormalize(n, nhat);

    A2D::Vec<T,3> t1;
    if constexpr (Data::has_ref_axis) {
        // shell ref axis transform
        t1 = A2D::Vec<T,3>(refAxis);
        
    } else { // doesn't have ref axis
        // shell natural transform
        t1 = A2D::Vec<T,3>(dXdxi);
    }

    // remove normal component from t1
    A2D::Vec<T,3> temp, t1hat;
    T d = A2D::VecDotCore<T,3>(nhat.get_data(), t1.get_data());
    A2D::VecSum(1.0, t1, -d, nhat, temp);
    A2D::VecNormalize(temp, t1hat);

    // compute t2 by cross product of normalized unit vectors
    A2D::Vec<T,3> t2hat;
    A2D::VecCross(nhat, t1hat, t2hat);

    // save values in Tmat 3x3 matrix
    for (int i = 0; i < 3; i++) {
        Tmat[3*i] = t1hat[i];
        Tmat[3*i+1] = t2hat[i];
        Tmat[3*i+2] = nhat[i];
    }
}

template <typename T, int vars_per_node, class Data, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrain(
    const T quad_pt[], const T refAxis[], const T xpts[], const T vars[], const T fn[], T et[]) {
    // TODO : do we actually need Ctn, Tn, XdinvTn, u0xn here?

    T etn[Basis::num_nodes];
    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        T pt[2];
        Basis::getNodePoint(inode,pt);

        // get shell transform and Xdn frame scope
        T Tmat[9], Xd[9];
        {
            // compute the computational coord gradients of Xpts for xi, eta
            T dXdxi[3], dXdeta[3];
            Basis::template interpFieldsGrad<3, 3>(pt, xpts, dXdxi, dXdeta);

            // assemble Xd frame
            Basis::assembleFrame(dXdxi, dXdeta, &fn[3*inode], Xd);
            
            // compute the shell transform based on the ref axis in Data object
            ShellComputeTransform<T, Data>(refAxis, dXdxi, dXdeta, &fn[3*inode], Tmat);
        } // end of Xd and shell transform scope

        // assemble u0xn frame scope
        T u0xn[9];
        {
            // compute midplane disp field gradients
            T u0xi[3], u0eta[3];
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, u0xi, u0eta);

            A2D::Vec<T,3> zero;
            Basis::assembleFrame(u0xi, u0eta, zero.get_data(), u0xn);
        }

        // compute rotation matrix at this node
        T C[9], tmp[9];
        Director::template computeRotationMat<vars_per_node, 1>(&vars[vars_per_node*inode], C);

        // compute Ct = T^T * C * T
        using MatOp = A2D::MatOp;
        A2D::MatMatMultCore3x3<T,MatOp::TRANSPOSE>(Tmat, C, tmp);
        A2D::MatMatMultCore3x3<T>(tmp, Tmat, C);

        // inverse Xd frame and Transformed product
        T XdinvTn[9];
        A2D::MatInvCore<T,3>(Xd, tmp);
        A2D::MatMatMultCore3x3<T>(tmp, Tmat, XdinvTn);

        // Compute transformation u0x = T^T * ueta * (Xdinv*T)
        A2D::MatMatMultCore3x3<T>(u0xn, XdinvTn, tmp);
        A2D::MatMatMultCore3x3<T,MatOp::TRANSPOSE>(Tmat, tmp, u0xn);

        // compute the drill strain
        etn[inode] = Director::evalDrillStrain(u0xn, C);

    } // end of node for loop

    // now interpolate to single et value
    Basis::template interpFields<1,1>(quad_pt, etn, et);

} // end of method ShellComputeDrillStrain

template <typename T, int vars_per_node, class Data, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainSens(
    const T quad_pt[], const T refAxis[], const T xpts[], const T vars[], const T fn[], const T et_bar[], T res[]) {
    // TODO : do we actually need Ctn, Tn, XdinvTn, u0xn here?

    // first interpolate back to nodal level
    T etn_bar[Basis::num_nodes];
    Basis::template interpFieldsTranspose<1, 1>(quad_pt, et_bar, etn_bar);

    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        T pt[2];
        Basis::getNodePoint(inode,pt);

        // get shell transform and Xdn frame scope
        T Tmat[9], Xd[9];
        {
            // compute the computational coord gradients of Xpts for xi, eta
            T dXdxi[3], dXdeta[3];
            Basis::template interpFieldsGrad<3, 3>(pt, xpts, dXdxi, dXdeta);

            // assemble Xd frame
            Basis::assembleFrame(dXdxi, dXdeta, &fn[3*inode], Xd);
            
            // compute the shell transform based on the ref axis in Data object
            T Tmat[9];
            ShellComputeTransform<T, Data>(refAxis, dXdxi, dXdeta, &fn[3*inode], Tmat);
        } // end of Xd and shell transform scope

        T u0xn_bar[9], C_bar[9];
        Director::evalDrillStrainSens(etn_bar[inode], u0xn_bar, C_bar);

        // backwards prop C2_bar to C_bar through T^t * C * T operation
        T tmp[9];
        {
            using MatOp = A2D::MatOp;
            A2D::MatMatMultCore3x3<T,MatOp::TRANSPOSE>(Tmat, C_bar, tmp);
            A2D::MatMatMultCore3x3<T>(tmp, Tmat, C_bar);
        }

        // backprop u0x_bar to u0xn_bar^T
        {
            T XdinvTn[9];
            A2D::MatInvCore<T,3>(Xd, tmp); // Xdinv
            A2D::MatMatMultCore3x3<T>(tmp, Tmat, XdinvTn);

            // reverse of u0x = T^t * u0xn * XdinvT is:
            // u0xn_bar = XdinvT^t * u0x_bar * T
            // but we want transpose version so each row is u0xi, u0eta, zero (formerly columns)
            // u0xn_bar^t = T^t * u0x_bar^t * XdinvT
            using MatOp = A2D::MatOp;
            A2D::MatMatMultCore3x3<T,MatOp::TRANSPOSE>(u0xn_bar, XdinvTn, tmp);
            A2D::MatMatMultCore3x3<T,MatOp::TRANSPOSE>(Tmat,tmp, u0xn_bar);
        }

        // reverse C(vars) to C_bar => res
        Director::template computeRotationMatSens<vars_per_node, 1>(C_bar, &res[vars_per_node*inode]);

        // reverse the interpolations u0xn_bar to res
        // because we have u0xn_bar^T stored, each row is u0xi_bar, u0eta_bar
        Basis::template interpFieldsGradTranspose<vars_per_node,3>(pt, &u0xn_bar[0], &u0xn_bar[3], res);

    } // end of node for loop
} // end of method ShellComputeDrillStrain