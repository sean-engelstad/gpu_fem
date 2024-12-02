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
    A2D::Vec<T,3> temp, t1hat, d;
    A2D::VecCross(nhat, t1, d);
    A2D::VecSum(1.0, t1, -1.0, d, temp);
    A2D::VecNormalize(temp, t1hat);

    // compute t2 by cross product of normalized unit vectors
    A2D::Vec<T,3> t2hat;
    A2D::VecCross(nhat, t1hat, t2hat);

    // save values in Tmat 3x3 matrix
    for (int i = 0; i < 3; i++) {
        T *_tmat = &Tmat[3*i];
        _tmat[0] = t1hat[i];
        _tmat[1] = t2hat[i];
        _tmat[2] = nhat[i];
    }
}

template <typename T, int vars_per_node, class Data, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrain(
    const Data physData, const T Xdn[], const T vars[], T etn[]) {
    // TODO : do we actually need Ctn, Tn, XdinvTn, u0xn here?

    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        T pt[2];
        Basis::getNodePoint(inode,pt);

        // get the computational gradients out of Xdn
        T dXdxi[3], dXdeta[3], n0[3];
        Basis::extractFrame(&Xdn[9*inode], dXdxi, dXdeta, n0);
        
        // compute the shell transform based on the ref axis in Data object
        T Tmat[9];
        ShellComputeTransform<T, Data>(physData.refAxis, dXdxi, dXdeta, n0, Tmat);
        
        // compute midplane disp field gradients
        T u0xi[3], u0eta[3];
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, u0xi, u0eta);

        // compute inverse of shell coordinate frame
        T Xdinv[9];
        A2D::MatInvCore<T,3>(&Xdn[9*inode], Xdinv);

        // compute XdinvT = Xdinv * T
        T XdinvTn[9], u0xn[9];
        A2D::MatMatMultCore3x3<T>(Xdinv, Tmat, XdinvTn);
        A2D::Vec<T,3> zero;
        Basis::assembleFrame(u0xi, u0eta, zero.get_data(), u0xn);

        // compute rotation matrix at this ndoe
        T C[9], tmp[9];
        Director::template computeRotationMat<vars_per_node, 1>(&vars[vars_per_node*inode], C);

        // compute Ct = T^T * C * T
        using MatOp = A2D::MatOp;
        A2D::MatMatMultCore3x3<T,MatOp::TRANSPOSE>(Tmat, C, tmp);
        A2D::MatMatMultCore3x3<T>(tmp, Tmat, C);

        // Compute transformation u0x = T^T * ueta * (Xdinv*T)
        A2D::MatMatMultCore3x3<T>(u0xn, XdinvTn, tmp);
        A2D::MatMatMultCore3x3<T,MatOp::TRANSPOSE>(Tmat, tmp, u0xn);

        // compute the drill strain
        etn[inode] = Director::evalDrillStrain(u0xn, C);

    } // end of node for loop
} // end of method ShellComputeDrillStrain