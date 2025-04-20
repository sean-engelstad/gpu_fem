#pragma once

#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#include "_utils.h"
#include "mesh/vtk_writer.h"

template <class Mat, class Vec>
using LinearSolveFunc = void (*)(Mat &, Vec &, Vec &, bool);

// assume data is on the device
template <typename T, class Mat, class Vec, class Assembler>
void newton_solve(LinearSolveFunc<Mat, Vec> linear_solve, Mat &kmat, Vec &loads, Vec &soln,
                  Assembler &assembler, Vec &res, Vec &rhs, Vec &vars, int num_load_factors,
                  T min_load_factor, T max_load_factor, int num_newton, T abs_tol, T rel_tol,
                  std::string outputFilePrefix, bool print = false) {
    for (int iload = 0; iload < num_load_factors; iload++) {
        T load_factor =
            min_load_factor + (max_load_factor - min_load_factor) * iload / (num_load_factors - 1);

        T init_res = 1e50;
        if (print) {
            printf("load step %d / %d : load factor %.4e\n", iload, num_load_factors, load_factor);
        }

        for (int inewton = 0; inewton < num_newton; inewton++) {
            // compute internal residual and stiffness matrix
            assembler.set_variables(vars);
            assembler.add_jacobian(res, kmat);
            assembler.apply_bcs(res);
            assembler.apply_bcs(kmat);

            // compute the RHS
            rhs.zeroValues();
            CUBLAS::axpy(load_factor, loads, rhs);
            CUBLAS::axpy(-1.0, res, rhs);
            assembler.apply_bcs(rhs);
            double rhs_norm = CUBLAS::get_vec_norm(rhs);

            // solve for the change in variables (soln = u - u0) and update variables
            soln.zeroValues();
            bool linear_print = false;
            linear_solve(kmat, rhs, soln, linear_print);
            double soln_norm = CUBLAS::get_vec_norm(soln);
            CUBLAS::axpy(1.0, soln, vars);

            // compute the residual (much cheaper computation on GPU)
            assembler.set_variables(vars);
            // assembler.add_residual(res); // TODO : for some reason using this add_residual
            // doesn't match add_jacobian res..
            assembler.add_jacobian(res, kmat);
            assembler.apply_bcs(res);
            rhs.zeroValues();
            CUBLAS::axpy(load_factor, loads, rhs);
            CUBLAS::axpy(-1.0, res, rhs);
            assembler.apply_bcs(rhs);
            double full_resid_norm = CUBLAS::get_vec_norm(rhs);
            // if (print) {
            //     printf("\t\tfull res = %.4e\n", full_resid_norm);
            // }
            if (inewton == 0) {
                init_res = full_resid_norm;
            }
            // TODO : need residual check
            if (print) {
                printf("\tnewton step %d, rhs = %.4e, soln = %.4e\n", inewton, full_resid_norm,
                       soln_norm);
            }

            if (abs(full_resid_norm) < (abs_tol + rel_tol * init_res)) {
                break;
            }
        }  // end of newton loop

        // write out solution
        auto h_vars = vars.createHostVec();
        std::stringstream outputFile;
        outputFile << outputFilePrefix << iload << ".vtk";
        printToVTK<Assembler, HostVec<T>>(assembler, h_vars, outputFile.str());

        // temp debug
        // if (print) {
        //     rhs.zeroValues();
        //     CUBLAS::axpy(load_factor, loads, rhs);
        //     auto h_vars2 = vars.createHostVec();
        //     auto h_rhs = rhs.createHostVec();
        //     int inode = 16; // 0-based, is 17 1-based
        //     printf("end z nodal force = %.4e\n", h_rhs[6*inode + 2]);
        //     printf("end z disp = %.4e\n", h_vars2[6*inode + 2]);

        //     inode = 33; // 0-based, is 17 1-based
        //     printf("end z nodal force 2 = %.4e\n", h_rhs[6*inode + 2]);
        //     printf("end z disp 2 = %.4e\n", h_vars2[6*inode + 2]);
        // }

    }  // end of load factor loop
}