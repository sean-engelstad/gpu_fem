#pragma once

template <class Data>
__HOST_DEVICE__ static void baseline_add_element_quadpt_tying_residual(
    const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
    const T vars[dof_per_elem], const Data physData, const T XdinvT[], const T &detXd,
    T res[dof_per_elem]) {  // const T fn[xpts_per_elem],
    /* add in-plane and transverse tying strain contributions from g11, g13 strains */
    if (!active_thread) return;

    // data to store in forwards + backwards section
    T quad_pt[2];
    T weight = Quadrature::getQuadraturePoint(iquad, quad_pt);
    static constexpr bool is_nonlinear = Phys::is_nonlinear;

    T fn[12];
    ShellComputeNodeNormals<T, Basis>(xpts, fn);

    // compute normals and directors
    T d[12];
    Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

    A2D::ADObj<A2D::SymMat<T, 3>> e0ty;

    {  // forward part
        T ety[9];
        computeTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);

        A2D::SymMat<T, 3> gty;
        interpTyingStrain<T, Basis>(quad_pt, ety, gty.get_data());
        A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty.value());
    }

    {
        auto &e0tyf = e0ty.value();

        // now compute in-plane and transverse tying strains
        {
            // first compute in-plane and transverse shear strains
            T e0[3], etr[2];
            e0[0] = e0tyf[0];         // e11
            e0[1] = e0tyf[3];         // e22
            e0[2] = 2.0 * e0tyf[1];   // e12
            etr[0] = 2.0 * e0tyf[4];  // e23, transverse shear
            etr[1] = 2.0 * e0tyf[2];  // e13, transverse shear

            // now compute the equiv stresses (as backprop strains)
            T e0b[3], etrb[2];
            {
                T eb[3], C[6];
                Data::evalTangentStiffness2D(physData.E, physData.nu, C);
                T thick = physData.thick;
                // in-plane A matrix term
                A2D::SymMatVecCoreScale3x3<T, false>(thick, C, e0, e0b);
                // skip B matrix term for now (assume thickOffset = 0)
                // transverse shear part
                T As = Data::getTransShearCorrFactor() * thick * C[5];
                etrb[1] = As * etr[1];
            }

            // now backprop back to e0tyb
            auto &e0tyb = e0ty.bvalue();
            e0tyb[0] = e0b[0];
            e0tyb[2] = 2.0 * etrb[1];
        }
    }  // end of physics part and backprop to gtyb

    A2D::Vec<T, 12> d_bar;
    {  // now backprop through frame rotation
        A2D::SymMat<T, 3> gty_bar;
        A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.bvalue().get_data(), gty_bar.get_data());

        T ety_bar[9];  // backprop unrotated tying strain
        interpTyingStrainTranspose<T, Basis>(quad_pt, gty_bar.get_data(), ety_bar);

        computeTyingStrainSens<T, Phys, Basis>(xpts, fn, vars, d, ety_bar, res, d_bar.get_data());
    }

    Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar.get_data(), res);
}

template <class Data>
__HOST_DEVICE__ static void add_element_quadpt_tying_residual(
    const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
    const T vars[dof_per_elem], const Data physData, const T XdinvT[], const T &detXd,
    T res[dof_per_elem]) {  // const T fn[xpts_per_elem],
    /* add in-plane and transverse tying strain contributions from g11, g13 strains */
    if (!active_thread) return;

    // data to store in forwards + backwards section
    T quad_pt[2];
    T weight = Quadrature::getQuadraturePoint(iquad, quad_pt);
    static constexpr bool is_nonlinear = Phys::is_nonlinear;

    // compute the forward tying strains
    A2D::SymMat<T, 3> gty;

    T fn[12];
    ShellComputeNodeNormals<T, Basis>(xpts, fn);

    // compute normals and directors
    T d[12];
    Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

    {  // compute and interp forward g11 and g13
        constexpr int num_tying_g11 = Basis::num_tying_points(0);
        T N_g11[num_tying_g11];  // same interp and points used for g13
        Basis::template getTyingInterp<0>(quad_pt, N_g11);

#pragma unroll
        for (int itying = 0; itying < num_tying_g11; itying++) {
            {
                T pt[2];
                Basis::template getTyingPoint<0>(itying, pt);

                // store g11 strain
                // gty[0] += N_g11[itying] * A2D::VecDotCore<T, 3>(Uxi, Xxi);
                gty[0] += N_g11[itying] *
                          Basis::template interpFieldsGradDotProduct<3, vars_per_node, 3, 0, 0>(
                              pt, xpts, vars);
                // store g13 strain
                // gty[2] +=
                //     N_g11[itying] *
                //     (0.5 * (A2D::VecDotCore<T, 3>(Xxi, d0) + A2D::VecDotCore<T, 3>(n0,
                //     Uxi)));
                // gty[2] += N_g11[itying] * 0.5 *
                //           Basis::template interpFieldsMixedDotProduct<3, 3, 3, 0>(pt, xpts,
                //           d);
                // gty[2] += N_g11[itying] * 0.5 *
                //           Basis::template interpFieldsMixedDotProduct<vars_per_node, 3, 3,
                //           0>(
                //               pt, vars, fn);

                // nonlinear g11 strain term
                if constexpr (is_nonlinear) {
                    // 0.5 * <Uxi, Uxi> => g11
                    gty[0] += N_g11[itying] * 0.5 *
                              Basis::template interpFieldsGradDotProduct<3, vars_per_node, 3, 0, 0>(
                                  pt, vars, vars);
                    // 0.5 * <Uxi, d0> => g13
                    gty[2] += N_g11[itying] * 0.5 *
                              Basis::template interpFieldsMixedDotProduct<vars_per_node, 3, 3, 0>(
                                  pt, vars, d);
                }
            }
        }
    }  // end of compute and interp forward g11 and g13

    {  // compute and interp forward g22 and g23
        constexpr int num_tying_g22 = Basis::num_tying_points(1);
        T N_g22[num_tying_g22];  // same interp and points used for g23
        Basis::template getTyingInterp<1>(quad_pt, N_g22);

#pragma unroll
        for (int itying = 0; itying < num_tying_g22; itying++) {
            {
                T pt[2];
                Basis::template getTyingPoint<1>(itying, pt);
                gty[3] += N_g22[itying] *
                          Basis::template interpFieldsGradDotProduct<3, vars_per_node, 3, 1, 1>(
                              pt, xpts, vars);
                // store g13 strain
                // gty[4] += N_g22[itying] * 0.5 *
                //           Basis::template interpFieldsMixedDotProduct<3, 3, 3, 1>(pt, xpts,
                //           d);
                // gty[4] += N_g22[itying] * 0.5 *
                //           Basis::template interpFieldsMixedDotProduct<vars_per_node, 3, 3,
                //           1>(
                //               pt, vars, fn);

                // nonlinear g11 strain term
                if constexpr (is_nonlinear) {
                    // 0.5 * <Ueta, Ueta> => g22
                    gty[3] += N_g22[itying] * 0.5 *
                              Basis::template interpFieldsGradDotProduct<3, vars_per_node, 3, 1, 1>(
                                  pt, vars, vars);
                    // 0.5 * <Ueta, d0> => g23
                    gty[4] += N_g22[itying] * 0.5 *
                              Basis::template interpFieldsMixedDotProduct<vars_per_node, 3, 3, 1>(
                                  pt, vars, d);
                }
            }
        }
    }  // end of compute and interp forward g22 and g23

    {  // compute and interp forward g12
        constexpr int num_tying_g12 = Basis::num_tying_points(2);
        T N_g12[num_tying_g12];  // same interp and points used for g23
        Basis::template getTyingInterp<2>(quad_pt, N_g12);

#pragma unroll
        for (int itying = 0; itying < num_tying_g12; itying++) {
            {
                T pt[2];
                Basis::template getTyingPoint<2>(itying, pt);
                T scal = 0.5 * N_g12[itying];
                gty[1] +=
                    scal * Basis::template interpFieldsGradDotProduct<3, vars_per_node, 3, 0, 1>(
                               pt, xpts, vars);
                gty[1] +=
                    scal * Basis::template interpFieldsGradDotProduct<3, vars_per_node, 3, 1, 0>(
                               pt, xpts, vars);

                // nonlinear g11 strain term
                if constexpr (is_nonlinear) {
                    // 0.5 * <Ueta, Ueta> => g22
                    gty[1] += N_g12[itying] * 0.5 *
                              Basis::template interpFieldsGradDotProduct<3, vars_per_node, 3, 0, 1>(
                                  pt, vars, vars);
                }
            }
        }
    }  // end of compute and interp forward g22 and g23

    T gtyb[6];  // backprop unrotated tying strain
    {
        A2D::ADObj<A2D::SymMat<T, 3>> e0ty;
        A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty.value());
        auto &e0tyf = e0ty.value();

        // now compute in-plane and transverse tying strains
        {
            // first compute in-plane and transverse shear strains
            T e0[3], etr[2];
            e0[0] = e0tyf[0];         // e11
            e0[1] = e0tyf[3];         // e22
            e0[2] = 2.0 * e0tyf[1];   // e12
            etr[0] = 2.0 * e0tyf[4];  // e23, transverse shear
            etr[1] = 2.0 * e0tyf[2];  // e13, transverse shear

            // now compute the equiv stresses (as backprop strains)
            T e0b[3], etrb[2];
            {
                T eb[3], C[6];
                Data::evalTangentStiffness2D(physData.E, physData.nu, C);
                T thick = physData.thick;
                // in-plane A matrix term
                A2D::SymMatVecCoreScale3x3<T, false>(thick, C, e0, e0b);
                // skip B matrix term for now (assume thickOffset = 0)
                // transverse shear part
                T As = Data::getTransShearCorrFactor() * thick * C[5];
                etrb[1] = As * etr[1];
            }

            // now backprop back to e0tyb
            auto &e0tyb = e0ty.bvalue();
            e0tyb[0] = e0b[0];
            e0tyb[2] = 2.0 * etrb[1];
        }

        // now backprop through frame rotation
        A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.bvalue().get_data(), gtyb);
    }  // end of physics part and backprop to gtyb

    A2D::Vec<T, 12> d_bar;
    // A2D::Vec<T, 3> d_bar;  // should be 12 not 3 (just testing)

    {  // reverse of g11 and g13 => res
        constexpr int num_tying_g11 = Basis::num_tying_points(0);
        T N_g11[num_tying_g11];
        Basis::template getTyingInterp<0>(quad_pt, N_g11);

#pragma unroll
        for (int itying = 0; itying < num_tying_g11; itying++) {
            // first entry for g11 reverse
            {
                T pt[2];
                Basis::template getTyingPoint<0>(itying, pt);  // huge register

                // asm volatile("" ::: "memory");

                {                                      // first for g13
                    T etyb = gtyb[2] * N_g11[itying];  // reversed interp from g13
                    // Basis::template interpFieldsMixedDotProduct_Sens<vars_per_node, 3, 3, 0>(
                    //     etyb, pt, res, xpts);  // fn (fn way slower than xpts..)
                    // Basis::template interpFieldsMixedDotProduct_Sens<vars_per_node, 3, 3, 0>(
                    //     etyb, pt, res, fn);  // fn (fn way slower than xpts..)
                    // Basis::template interpFieldsGradDotProduct_RightSens<3, vars_per_node, 3,
                    // 0,
                    //                                                      0>(etyb, pt, xpts,
                    //                                                         d_bar.get_data());
                }

                // {                                      // first for g11
                //     T etyb = gtyb[0] * N_g11[itying];  // reversed interp from g11
                //     Basis::template interpFieldsGradDotProduct_RightSens<3, vars_per_node, 3,
                //     0,
                //                                                          0>(etyb, pt, xpts,
                //                                                             res);
                // }

                // nonlinear g11 strain term backprop
                // if constexpr (is_nonlinear) {
                //     A2D::VecAddCore<T, 3>(etyb, Uxi, Uxi_bar.get_data());
                // }

                // Basis::template interpFieldsGradTransposeLight<vars_per_node, 3>(
                //     pt, Uxi_bar.get_data(), zero.get_data(), res);
            }
        }

        // Director::template computeDirectorSens<vars_per_node, num_nodes>(fn,
        // d_bar.get_data(),
        //                                                                  res);
    }  // end of g11 reverse
}