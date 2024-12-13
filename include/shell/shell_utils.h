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
    A2D::VecSum(T(1.0), t1, -d, nhat, temp);
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

        // Compute transformation u0x = T^T * u0xn * (Xdinv*T)
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
    A2D::Vec<T,Basis::num_nodes> etn_bar;
    Basis::template interpFieldsTranspose<1, 1>(quad_pt, et_bar, etn_bar.get_data());

    constexpr A2D::MatOp NORM = A2D::MatOp::NORMAL;
    constexpr A2D::MatOp TRANS = A2D::MatOp::TRANSPOSE;

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

        T u0xn_bar[9], C_bar[9]; // really u0x_bar at this point (but don't want two vars for it)
        Director::evalDrillStrainSens(etn_bar[inode], u0xn_bar, C_bar);

        T tmp[9];
        {
            // C_bar = T * C2_bar * T^t (in reverse)
            A2D::MatMatMultCore3x3<T>(Tmat, C_bar, tmp);
            A2D::MatMatMultCore3x3<T,NORM,TRANS>(tmp, Tmat, C_bar);

            // reverse C(vars) to C_bar => res
            Director::template computeRotationMatSens<vars_per_node, 1>(C_bar, &res[vars_per_node*inode]);
        }

        // backprop u0x_bar to u0xn_bar^T
        {
            T XdinvT[9];
            A2D::MatInvCore<T,3>(Xd, tmp); // Xdinv
            A2D::MatMatMultCore3x3<T>(tmp, Tmat, XdinvT);

            // u0xn_bar = T * u0x_bar * XdinvT^t
            // transpose version for convenience of cols avail in rows
            // u0xn_bar^t = XdinvT * u0x_bar^t * T^t (u0xn_bar now holds transpose u0xn_bar^t)
            A2D::MatMatMultCore3x3<T,NORM,TRANS>(XdinvT, u0xn_bar, tmp);
            A2D::MatMatMultCore3x3<T,NORM,TRANS>(tmp, Tmat, u0xn_bar);

            // reverse the interpolations u0xn_bar to res
            // because we have u0xn_bar^T stored, each row is u0xi_bar, u0eta_bar
            Basis::template interpFieldsGradTranspose<vars_per_node,3>(pt, &u0xn_bar[0], &u0xn_bar[3], res);
        }

    } // end of node for loop
} // end of method ShellComputeDrillStrain