template <typename T2>
__HOST_DEVICE__ static void computeQuadptSectionalLoads(const Data &physData, const T &scale,
                                                        A2D::Mat<T2, 3, 3> &u0x,
                                                        A2D::Mat<T2, 3, 3> &u1x,
                                                        A2D::SymMat<T2, 3> &e0ty,
                                                        A2D::Vec<T2, 1> &et, A2D::Vec<T, 9> S) {
    A2D::ADObj<A2D::Vec<T, 9>> E;

    // use stack to compute shell strains, stresses and then to strain energy
    auto strain_energy_stack =
        A2D::MakeStack(A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
                       A2D::IsotropicShellStress<T, Data>(physData.E, physData.nu, physData.thick,
                                                          physData.tOffset, E, S));

    // technically the IsotropicShellStress computes sectional stresses aka sectional loads
}  // end of computeQuadptSectionalLoads

template <typename T2>
__HOST_DEVICE__ static void computeQuadptStrains(const Data &physData, const T &scale,
                                                 A2D::Mat<T2, 3, 3> &u0x, A2D::Mat<T2, 3, 3> &u1x,
                                                 A2D::SymMat<T2, 3> &e0ty, A2D::Vec<T2, 1> &et,
                                                 A2D::Vec<T, 9> E) {
    // use stack to compute shell strains, stresses and then to strain energy
    auto strain_energy_stack = A2D::MakeStack(A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E));
}  // end of computeQuadptStrains

template <typename T2>
__HOST_DEVICE__ static void computeQuadptStrainsSens(const Data &physData, const T &scale,
                                                     A2D::ADObj<A2D::Mat<T2, 3, 3>> &u0x,
                                                     A2D::ADObj<A2D::Mat<T2, 3, 3>> &u1x,
                                                     A2D::ADObj<A2D::SymMat<T2, 3>> &e0ty,
                                                     A2D::ADObj<A2D::Vec<T2, 1>> &et,
                                                     T strain_bar[9]) {
    A2D::Vec<T, 9> E;
    // use stack to compute shell strains, stresses and then to strain energy
    auto strain_energy_stack = A2D::MakeStack(A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E));
    T *Eb = E.bvalue();
    for (int i = 0; i < 6; i++) {
        Eb[i] = strain_bar[i];
    }
    strain_energy_stack.reverse();
}  // end of computeQuadptStrains

__HOST_DEVICE__
static void computeKSFailureSVSens(const Data &data, T rho_KS, const T strains[vars_per_node],
                                   const T fail_index_bar, T strain_bar[vars_per_node]) {
    // compute the von mises ksfailure index in the shell
    // strains are the 'nodal_strains' 6 of them
    // x_bar is a scalar

    // upper and lower surface thicknesses in z the normal coordinate
    T zU = (0.5 - data.tOffset) * data.thick;
    T zL = -(0.5 + data.tOffset) * data.thick;

    // compute the upper and lower surface strains
    T strainU[3], strainL[3];
    A2D::VecSumCore<T, 3>(1.0, &strains[0], zU, &strains[3], strainU);
    A2D::VecSumCore<T, 3>(1.0, &strains[0], zL, &strains[3], strainL);

    // compute the 2D tangent stiffness matrix C or Q
    T C[6];
    Data::evalTangentStiffness2D(data.E, data.nu, C);

    // compute upper and lower surface stresses
    T stressU[3], stressL[3];
    A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, strainL, stressL);
    A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, strainU, stressU);

    // note stress[3] refers to s11, s22, s12 stresses
    // compute von mises failure at upper and lower surfaces
    T failU, failL;
    vonMisesFailure2D(data, stressU, &failU);
    vonMisesFailure2D(data, stressL, &failL);

    // now do KS max among upper and lower surface stresses
    T ksMax;
    if (failU > failL) {
        ksMax = failU;
    } else {
        ksMax = failL;
    }

    T ksSum = exp(rho_KS * (failU - ksMax)) + exp(rho_KS * (failL - ksMax));
    T ksVal = ksMax + log(ksSum) / rho_KS;

    // end of forward analysis
    // backprop now first to dsigma_KS/dthick through zL and zU

    // first for lower stresses
    T fail_bar = exp(rho_KS * (failL - ksMax)) / ksSum;
    T stress_bar[3];
    vonMisesFailure2DSens(data, stressL, fail_bar, stress_bar);
    T surf_strain_bar[3];
    A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, stress_bar, surf_strain_bar);

    // backprop from lower surf strains to strain_bar
    A2D::VecSumCore<T, 3>(surf_strain_bar, &strain_bar[0]);
    A2D::VecSumCore<T, 3>(zL, surf_strain_bar, &strain_bar[3]);

    // then for upper stresses
    fail_bar = exp(rho_KS * (failU - ksMax)) / ksSum;
    vonMisesFailure2DSens(data, stressU, fail_bar, stress_bar);
    A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, stress_bar, surf_strain_bar);

    // backprop from upper surf strains to strain_bar
    A2D::VecSumCore<T, 3>(surf_strain_bar, &strain_bar[0]);
    A2D::VecSumCore<T, 3>(zU, surf_strain_bar, &strain_bar[3]);

    return ksVal;
}

__HOST_DEVICE__ static void compute_dRdx(const int local_dv, const Data &physData, const T &scale,
                                         A2D::ADObj<A2D::Mat<T, 3, 3>> &u0x,
                                         A2D::ADObj<A2D::Mat<T, 3, 3>> &u1x,
                                         A2D::ADObj<A2D::SymMat<T, 3>> &e0ty,
                                         A2D::ADObj<A2D::Vec<T, 1>> &et) {
    // for isotropic material only one local_dv aka thickness
    // so ignore this input

    using T2 = A2D::ADScalar<T, 1>;

    // copy into local
    A2D::ADObj<A2D::Mat<T2, 3, 3>> _u0x, _u1x;
    A2D::ADObj<A2D::SymMat<T2, 3>> _e0ty;
    A2D::ADObj<A2D::Vec<T2, 1>> _et;

    for (int i = 0; i < 9; i++) {
        _u0x.value()[i].value = u0x.value()[i];
        _u1x.value()[i].value = u1x.value()[i];
    }
    for (int i = 0; i < 6; i++) {
        _e0ty.value()[i].value = e0ty.value()[i];
    }
    _et.value()[0].value = et.value()[0];

    // use forward AD to get dstrain/dthick?
    A2D::ADObj<A2D::Vec<T2, 9>> E, S;
    A2D::ADObj<T2> ES_dot, Uelem;

    // some way to check that T2 is indeed ADScalar<> type?
    T2 thick;
    if (local_dv == 0) {
        thick = physData.thick;
        thick.deriv = 1.0;
    }

    // use stack to compute shell strains, stresses and then to strain energy
    auto strain_energy_stack =
        A2D::MakeStack(A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
                       A2D::IsotropicShellStress<T2, Data>(T2(physData.E), T2(physData.nu), thick,
                                                           physData.tOffset, E, S),
                       // no 1/2 here to match TACS formulation (just scales eqns) [is removing
                       // the 0.5 correct?]
                       A2D::VecDot(E, S, ES_dot), A2D::Eval(T2(scale) * ES_dot, Uelem));
    // printf("Uelem = %.8e\n", Uelem.value());

    Uelem.bvalue() = 1.0;
    strain_energy_stack.reverse();
    // bvalue outputs stored in u0x, u1x, e0ty, et and are backpropagated

    // now copy back data from ADScalar to regular types
    for (int i = 0; i < 9; i++) {
        u0x.value()[i] = _u0x.value()[i].deriv;
        u1x.value()[i] = _u1x.value()[i].deriv;
    }
    for (int i = 0; i < 6; i++) {
        e0ty.value()[i] = _e0ty.value()[i].deriv;
    }
    et.value()[0] = _et.value()[0].deriv;
}  // end of computeWeakRes

template <class Data>
__HOST_DEVICE__ static void compute_element_quadpt_adj_res_product2(
    const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
    const T vars[dof_per_elem], const Data physData, const T psi[dof_per_elem], T loc_dv_sens[])

{
    /* 2nd version (not using hfwd) of adjoint residual product*/
    if (!active_thread) return;

    // data to store in forwards + backwards section
    T fn[3 * num_nodes];  // node normals
    T pt[2];              // quadrature point
    T d[3 * num_nodes];   // needed for reverse mode, nonlinear case
    T weight = Quadrature::getQuadraturePoint(iquad, pt);

    T res[24];
    memset(res, 0.0, 24 * sizeof(T));

    // in-out of forward & backwards section
    A2D::ADObj<A2D::Mat<T, 3, 3>> u0x, u1x;
    A2D::ADObj<A2D::SymMat<T, 3>> e0ty;
    A2D::ADObj<A2D::Vec<T, 1>> et;
    static constexpr bool is_nonlinear = Phys::is_nonlinear;

    // forward scope block for strain energy
    // ------------------------------------------------
    {
        // compute node normals fn
        ShellComputeNodeNormals<T, Basis>(xpts, fn);

        // compute the interpolated drill strain
        ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
            pt, physData.refAxis, xpts, vars, fn, et.value().get_data());

        // compute directors
        Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

        // compute tying strain
        T ety[Basis::num_all_tying_points];
        computeTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);

        // compute all shell displacement gradients
        T detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
            pt, physData.refAxis, xpts, vars, fn, d, ety, u0x.value().get_data(),
            u1x.value().get_data(), e0ty.value());

        // get the scale for disp grad sens of the energy
        T scale = detXd * weight;

        // compute energy + energy-dispGrad sensitivites with physics
        Phys::template computeWeakResThickDVSens<T>(physData, scale, u0x, u1x, e0ty, et);

    }  // end of forward scope block for strain energy
    // ------------------------------------------------

    auto u1xF = u0x.value();
    printf("u1xF:");
    printVec<T>(9, u1xF.get_data());
    auto u1xB = u1x.bvalue();
    printf("u1xb:");
    printVec<T>(9, u1xB.get_data());

    // beginning of backprop section to final residual derivatives
    // -----------------------------------------------------

    // compute disp grad sens u0x_bar, u1x_bar, e0ty_bar => res, d_bar,
    // ety_bar
    A2D::Vec<T, 3 * num_nodes> d_bar;
    A2D::Vec<T, Basis::num_all_tying_points> ety_bar;
    // T ety_bar[Basis::num_all_tying_points];
    ShellComputeDispGradSens<T, vars_per_node, Basis, Data>(
        pt, physData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(), u1x.bvalue().get_data(),
        e0ty.bvalue(), res, d_bar.get_data(), ety_bar.get_data());

    // backprop tying strain sens ety_bar to d_bar and res
    computeTyingStrainSens<T, Phys, Basis>(xpts, fn, vars, d, ety_bar.get_data(), res,
                                           d_bar.get_data());

    // directors back to residuals
    Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar.get_data(), res);

    // drill strain sens
    ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
        pt, physData.refAxis, xpts, vars, fn, et.bvalue().get_data(), res);

    // TODO : rotation constraint sens for some director classes (zero for
    // linear rotations

    printf("res_HC:");
    printVec<T>(6, res);

    loc_dv_sens[0] = A2D::VecDotCore<T, 24>(res, psi);

}  // end of method add_element_quadpt_residual

template <class Data>
__HOST_DEVICE__ static void compute_element_quadpt_strains_SVsens(
    const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
    const T vars[dof_per_elem], const Data physData, const T strain_bar[vars_per_node],
    T vars_bar[dof_per_elem])

{
    // keep in mind max of ~256 floats on single thread

    if (!active_thread) return;

    // data to store in forwards + backwards section
    T fn[3 * num_nodes];  // node normals
    T pt[2];              // quadrature point
    T weight = Quadrature::getQuadraturePoint(iquad, pt);

    // in-out of forward & backwards section
    A2D::ADObj<A2D::Mat<T, 3, 3>> u0x, u1x;
    A2D::ADObj<A2D::SymMat<T, 3>> e0ty;
    A2D::ADObj<A2D::Vec<T, 1>> et;
    A2D::ADObj<A2D::Vec<T, 9>> E;
    T d[3 * num_nodes];
    static constexpr bool is_nonlinear = Phys::is_nonlinear;

    // forward scope block for strain energy
    // ------------------------------------------------
    {
        // compute node normals fn
        ShellComputeNodeNormals<T, Basis>(xpts, fn);

        // compute the interpolated drill strain
        ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
            pt, physData.refAxis, xpts, vars, fn, et.value().get_data());

        // compute directors
        Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

        // compute tying strain
        T ety[Basis::num_all_tying_points];
        computeTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);

        // compute all shell displacement gradients
        T detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
            pt, physData.refAxis, xpts, vars, fn, d, ety, u0x.value().get_data(),
            u1x.value().get_data(), e0ty.value().get_data());

        // get the scale for disp grad sens of the energy
        T scale = detXd * weight;

        // compute energy + energy-dispGrad sensitivites with physics
        Phys::template computeQuadptStrainsSens<T>(physData, scale, u0x, u1x, e0ty, et, strain_bar);

    }  // end of forward scope block for strain energy
    // ------------------------------------------------

    // now we have u0x, u1x, e0ty, et bvalue's
    // we need to backprop to vars bvalue()
    // same code here as residual

    // compute disp grad sens u0x_bar, u1x_bar, e0ty_bar => res, d_bar,
    // ety_bar
    A2D::Vec<T, 3 * num_nodes> d_bar;
    A2D::Vec<T, Basis::num_all_tying_points> ety_bar;
    // T ety_bar[Basis::num_all_tying_points];
    ShellComputeDispGradSens<T, vars_per_node, Basis, Data>(
        pt, physData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(), u1x.bvalue().get_data(),
        e0ty.bvalue(), vars_bar, d_bar.get_data(), ety_bar.get_data());

    // drill strain sens
    ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
        pt, physData.refAxis, xpts, vars, fn, et.bvalue().get_data(), vars_bar);

    // backprop tying strain sens ety_bar to d_bar and res
    computeTyingStrainSens<Phys, Basis>(xpts, fn, vars, d, ety_bar.get_data(), vars_bar,
                                        d_bar.get_data());

    // directors back to residuals
    Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar.get_data(),
                                                                     vars_bar);
}