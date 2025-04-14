#include <iostream>

#include "chrono"
#include "linalg/linalg.h"
#include "mesh/vtk_writer.h"

#include <sstream>
#include <string>

// shell imports
#include "assembler.h"
#include "shell/shell_elem_group.h"
#include "shell/physics/isotropic_shell.h"

/**
 solve on CPU with cusparse for debugging
 **/

int main(void) {
    using T = double;

    std::ios::sync_with_stdio(false);  // always flush print immediately

    TACSMeshLoader<T> mesh_loader{};
    mesh_loader.scanBDFFile("Beam.bdf");

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = true;
    // constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    // material & thick properties
    double E = 1.2e6, nu = 0.0, thick = 0.1; 
    auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

    // check bcs
    // DeviceVec<int> d_bcs = assembler.getBCs();
    // auto h_bcs = d_bcs.createHostVec();
    // printf("bcs:");
    // printVec<int>(h_bcs.getSize(), h_bcs.getPtr());

    // perform a factorization on the rowPtr, colPtr (before creating matrix)
    double fillin = 10.0;  // 10.0
    assembler.symbolic_factorization(fillin, true);

    // compute load magnitude of tip force
    double length = 10.0; 
    double width = 1.0;
    double Izz = width * thick * thick * thick / 12.0;
    double beam_tip_force = 4.0 * E * Izz / length / length;

    // find nodes within tolerance of x=10.0
    int num_nodes = assembler.get_num_nodes();
    int num_vars = assembler.get_num_vars();
    HostVec<T> h_loads(num_vars);
    DeviceVec<T> d_xpts = assembler.getXpts();
    auto h_xpts = d_xpts.createHostVec();
    int num_tip_nodes = 0;
    for (int inode = 0; inode < num_nodes; inode++) {
        if (abs(h_xpts[3*inode] - length) < 1e-6) {
            num_tip_nodes++;
        }
    }
    for (int inode = 0; inode < num_nodes; inode++) {
        if (abs(h_xpts[3*inode] - length) < 1e-6) {
            h_loads[6*inode+2] = beam_tip_force / num_tip_nodes;
        }
    }
    auto d_fa = h_loads.createDeviceVec();
    assembler.apply_bcs(d_fa);

    double loads_norm = CUSPARSE::get_vec_norm(d_loads);
    printf("loads_norm = %.4e\n", loads_norm);


    // setup kmat, res, variables
    auto res = assembler.createVarsVec();
    auto rhs = assembler.createVarsVec();
    auto soln = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);

    T beta = 10.0;
    // T beta = 3.0;
    int nn = 32;
    int sym = -1; // no symmetry yet I believe
    double Hreg = 1e-4; // regularization for H
    DeviceVec<T> d_xs0 = assembler.getXpts();
    auto d_xa0 = d_xs0.createDeviceVec(); // copied for now
    // TODO : get the initial aero and struct meshes here..
    auto meld = MELD<T>(d_xs0, d_xa0, beta, nn, sym, Hreg);
    meld.initialize();


    // nonlinear static solve settings
    // -------------------------------
    // int num_load_steps = 1;
    int num_load_steps = 20;
    int num_newton = 30; //30;
    // double max_load_factor = 0.05;
    double max_load_factor = 1.0;

    // continuation solver
    // -------------------
    
    // outer aeroelastic coupling loop
    for (int iouter = 0; iouter < 5; iouter++) {
        
        double prev_aero_load_factor = iouter/5;
        double aero_load_factor = (iouter + 1)/5;

        // prev u_s => ua
        auto d_ua = meld.transferDisps(variables);

        // would be aero solve here modifying d_loads but not doing that here
        // maybe do load scaling here
        // flowSolver.solve(d_ua); // or iterate
        // d_fa = flowSolver.getAeroLoads();

        // then transfer aero loads to struct loads
        auto d_fs = meld.transferLoads(d_fa); // so is just the same thing here

        // inner nonlinear static solve (to be a solver class)
        for (int load_step = 0; load_step < num_load_steps; load_step++) {
            double load_factor = (aero_load_factor - prev_aero_load_factor) * (load_step + 1) / num_load_steps + prev_aero_load_factor;
            printf("load step %d, load_Factor = %.4e\n", load_step, load_factor);

            // Newton iteration nonlinear solve
            // --------------------------------
            for (int inewton = 0; inewton < num_newton; inewton++) {
                
                // compute internal residual and stiffness matrix
                assembler.set_variables(vars);
                assembler.add_jacobian(res, kmat);
                assembler.apply_bcs(res);
                assembler.apply_bcs(kmat);

                // need more continuation here..
                // look at Ali's restart method..

                // compute the RHS
                rhs.zeroValues();
                CUSPARSE::axpy(load_factor, d_fs, rhs);
                CUSPARSE::axpy(-1.0, res, rhs);
                assembler.apply_bcs(rhs);
                double rhs_norm = CUSPARSE::get_vec_norm(rhs);

                // printf("rhs[104] = %.4e\n", rhs[104]);

                // solve for the change in variables (soln = u - u0) and update variables
                soln.zeroValues();
                CUSPARSE::direct_LU_solve_old<T>(kmat, rhs, soln);
                double soln_norm = CUSPARSE::get_vec_norm(soln);
                printf("\tnewton step %d, rhs = %.4e, soln = %.4e\n", inewton, rhs_norm, soln_norm);
                CUSPARSE::axpy(1.0, soln, vars);


                // compute the residual (much cheaper computation on GPU)
                assembler.set_variables(vars);
                assembler.add_residual(res);
                rhs.zeroValues();
                CUSPARSE::axpy(load_factor, d_loads, rhs);
                CUSPARSE::axpy(-1.0, res, rhs);
                assembler.apply_bcs(rhs);
                double full_resid_norm = CUSPARSE::get_vec_norm(rhs);
                printf("\t\tfull res = %.4e\n", full_resid_norm);
                // TODO : need residual check

                if (abs(full_resid_norm) < 1e-7) {
                    break;
                }
            }

            std::stringstream filename;
            filename << "out/beam_" << load_step << ".vtk";

            auto h_vars = vars.createHostVec();
            printToVTK<Assembler, HostVec<T>>(assembler, h_vars, filename.str());
            
        } // end of nonlinear static analysis

        // then need some convergence criterion here..

    } // end of aeroelastic loop

    return 0;
};