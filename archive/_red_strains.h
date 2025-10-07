template <class Data>
    __HOST_DEVICE__ static void add_elem_drill_strain_jacobian_col(
        const bool active_thread, const int iquad, const int ivar, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, T matCol[dof_per_elem]) {
        
        /* compute drill strain contributions to the jacobian */
        if (!active_thread) return;

        T fn[3 * num_nodes], pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        static constexpr bool is_nonlinear = Phys::is_nonlinear;
        
        A2D::A2DObj<A2D::Vec<T, 1>> et;

        /* pre-compute steps */
        ShellComputeNodeNormals<T, Basis>(xpts, fn);

        /* regular forward part, nonlinear only*/
        if constexpr (is_nonlinear) {
            // forward part (only needed for nonlinear)
            ShellComputeDrillStrainFast<T, vars_per_node, Data, Basis, Director>(
                    pt, physData.refAxis, xpts, vars, fn, et.value().get_data());
        }

        /* forward deriv part, pvalues() */
        A2D::Vec<T, dof_per_elem> p_vars;
        p_vars[ivar] = 1.0;  // p_vars is unit vector for current column to compute
        // new method that doesn't compute for all nodes first (less compute)
        ShellComputeDrillStrainFastHfwd<T, vars_per_node, Data, Basis, Director>(
                    pt, physData.refAxis, xpts, p_vars, fn, et.pvalue().get_data());

        // get the scale for disp grad sens of the energy
        T detXd = getDetXd<T, Basis>(pt, xpts, fn);
        T scale = detXd * weight;

        /* need shorter weak forms here.. */
        // if just drill strain nonzero.. (A2D literals should only do that part, not need other memories)
        // maybe null objects
        Phys::template computeWeakJacobianCol<T>(physData, scale, null, null, null, et);

        /* breverse (1st order reverse), only needed for nonlinear case */
        if constexpr (is_nonlinear) {
            // res as nullptr?
            ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
                    pt, physData.refAxis, xpts, vars, fn, et.bvalue().get_data(), nullptr);
        }

        /* hreverse (2nd order proj hessian reverse) */
        ShellComputeDrillStrainHrev<T, vars_per_node, Data, Basis, Director>(
                    pt, physData.refAxis, xpts, vars, fn, et.hvalue().get_data(), matCol);
    }

    template <class Data>
    __HOST_DEVICE__ static void add_elem_bending_strain_jacobian_col(
        const bool active_thread, const int iquad, const int ivar, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, T matCol[dof_per_elem]) {
        
        /* compute bending strain contributions to the jacobian */
        if (!active_thread) return;

            // data to store in forwards + backwards section
            T fn[3 * num_nodes];  // node normals
            T pt[2];              // quadrature point
            T d[3 * num_nodes];   // need directors in reverse for nonlinear strains
            T weight = Quadrature::getQuadraturePoint(iquad, pt);
            static constexpr bool is_nonlinear = Phys::is_nonlinear;

            // in-out of forward & backwards section
            A2D::A2DObj<A2D::Mat<T, 3, 3>> u0x, u1x;

            // prelim block
            ShellComputeNodeNormals<T, Basis>(xpts, fn);

            // forward section
            if constexpr (is_nonlinear) {
                Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

                // TODO : don't store XdinvT here? and maybe can make faster bending disp grad code?
                T XdinvT[9];
                computeBendingDispGrad<T, vars_per_node, Basis, Data>(
                    pt, physData.refAxis, xpts, vars, fn, d, XdinvT, u0x.value().get_data(),
                    u1x.value().get_data());

            }  // end of forward scope

            // hforward section (pvalue's)
            A2D::Vec<T, dof_per_elem> p_vars;
            p_vars[ivar] = 1.0;  // p_vars is unit vector for current column to compute
            T p_d[3 * num_nodes];
            {
                Director::template computeDirectorHfwd<vars_per_node, num_nodes>(p_vars.get_data(), fn,
                                                                                p_d);

                // forward derivs of bending strains
                T XdinvT[9];
                computeBendingDispGradHfwd<T, vars_per_node, Basis, Data>(
                    pt, physData.refAxis, xpts, p_vars.get_data(), fn, p_d, 
                    XdinvT, u0x.pvalue().get_data(), u1x.pvalue().get_data());

            }  // end of hforward scope

            // derivatives over disp grad to strain energy portion
            // ---------------------
            T detXd = getDetXd<T, Basis>(pt, xpts, fn);
            T scale = detXd * weight;

            // TODO : how to make it do less compute for null cases..
            Phys::template computeWeakJacobianCol<T>(physData, scale, u0x, u1x, null, null);
            // ---------------------
            // begin reverse blocks from strain energy => physical disp grad sens

            // breverse (1st order derivs)
            if constexpr (is_nonlinear) {
                A2D::Vec<T, 3 * num_nodes> d_bar;  // zeroes out on init
                T XdinvT[9];
                computeBendingDispGradSens<T, vars_per_node, Basis, Data>(
                    pt, physData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(),
                    u1x.bvalue().get_data(), XdinvT, nullptr, d_bar.get_data());

                // Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar.get_data(),
                //                                                                  nullptr);

            }  // end of breverse scope (1st order derivs)

            // hreverse (2nd order derivs)
            {
                A2D::Vec<T, 3 * num_nodes> d_hat;                  // zeroes out on init
                T XdinvT[9];
                computeBendingDispGradHrev<T, vars_per_node, Basis, Data>(
                    pt, physData.refAxis, xpts, vars, fn, u0x.hvalue().get_data(),
                    u1x.hvalue().get_data(), XdinvT, matCol, d_hat.get_data());

                Director::template computeDirectorHrev<vars_per_node, num_nodes>(fn, d_hat.get_data(),
                                                                                matCol);

            }  // end of hreverse scope (2nd order derivs)
    }

    template <class Data>
    __HOST_DEVICE__ static void add_elem_tying_strain_jacobian_col(
        const bool active_thread, const int iquad, const int ivar, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, T matCol[dof_per_elem]) {
        
        if (!active_thread) return;

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T d[3 * num_nodes];   // need directors in reverse for nonlinear strains
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // printf("\tquadpt %d: (%.2e, %.2e) and weight %.2e\n", iquad, pt[0], pt[1], weight);

        // in-out of forward & backwards section
        A2D::A2DObj<A2D::SymMat<T, 3>> e0ty;

        ShellComputeNodeNormals<T, Basis>(xpts, fn);
        T XdinvT[9];
        computeXdinvT(pt, physData.refAxis, xpts, fn, XdinvT); // TODO : make this method

        // forward section
        if constexpr (is_nonlinear) {
            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // compute tying strain
            A2D::SymMat<T, 3> gty;
            computeFullTyingStrain<T, Phys, Basis, is_nonlinear>(pt, xpts, fn, vars, d, gty.get_data());

            // rotate the tying strains with XdinvT frame
            A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty.value());
        }  // end of forward scope

        // hforward section (pvalue's)
        A2D::Vec<T, dof_per_elem> p_vars;
        p_vars[ivar] = 1.0;  // p_vars is unit vector for current column to compute
        T p_d[3 * num_nodes];
        {
            Director::template computeDirectorHfwd<vars_per_node, num_nodes>(p_vars.get_data(), fn,
                                                                                p_d);

            // compute tying strain
            A2D::SymMat<T, 3> p_gty;
            computeFullTyingStrainHfwd<T, Phys, Basis>(pt, xpts, fn, vars, d, p_vars.get_data(), p_d, p_gty.get_data());

            // rotate the tying strains with XdinvT frame
            A2D::SymMatRotateFrame<T, 3>(XdinvT, p_gty, e0ty.pvalue());

        }  // end of hforward scope

        // derivatives over disp grad to strain energy portion
        // ---------------------
        T detXd = getDetXd<T, Basis>(pt, xpts, fn);
        T scale = detXd * weight;

        Phys::template computeWeakJacobianCol<T>(physData, scale, null, null, e0ty, null);
        // ---------------------
        // begin reverse blocks from strain energy => physical disp grad sens

        // breverse (1st order derivs)
        A2D::SymMat<T,3> gty_bar;
        if constexpr (is_nonlinear) {
            // transpose rotate the tying strains (frame transform)
            A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.bvalue().get_data(), gty_bar.get_data());

            Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar.get_data(),
                                                                                nullptr); // TODO : to have nullptr here?
        }  // end of breverse scope (1st order derivs)

        // hreverse (2nd order derivs)
        {
            A2D::Vec<T, 3 * num_nodes> d_hat;                  // zeroes out on init

            // transpose rotate the tying strains
            A2D::SymMat<T,3> gty_hat;
            A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.hvalue().get_data(), gty_hat.get_data());

            // backprop tying strain sens
            computeFullTyingStrainHrev<T, Phys, Basis>(pt, xpts, fn, vars, d, p_vars.get_data(), 
                p_d, gty_bar.get_data(), gty_hat.get_data(), matCol, d_hat.get_data());

            Director::template computeDirectorHrev<vars_per_node, num_nodes>(fn, d_hat.get_data(),
                                                                                matCol);
        }  // end of hreverse scope (2nd order derivs)

    }      // add_element_quadpt_jacobian_col