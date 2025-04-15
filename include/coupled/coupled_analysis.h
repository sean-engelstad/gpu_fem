#pragma once

// TODO : finish FuntofemCoupledAnalysis solver
// TODO : make an Mphys version as well (so we can compare)

template <typename T, class Vec, class StructSolver, class AeroSolver, class Transfer>
class FuntofemCoupledAnalysis {
    // an aeroelastic coupled analysis, later can change it to include other couplings if desired
   public:
    FuntofemCoupledAnalysis(StructSolver struct_solver, AeroSolver aero_solver, Transfer transfer,
                            int num_coupled_steps)
        : struct_solver(struct_solver),
          aero_solver(aero_solver),
          transfer(transfer),
          num_coupled_steps(num_coupled_steps) {
        // initialize coupled states
        ns = struct_solver.get_num_nodes();
        na = aero_solver.get_num_surf_nodes();
        us = Vec(6 * ns);
        ua = Vec(6 * na);
        fs = Vec(6 * ns);
        fa = Vec(6 * na);
    }

    void solve_forward() {
        // TODO : add optional uncoupled phase here?

        us.zeroValues();  // initial zero struct disps to start loop

        for (int icoupled = 0; icoupled < num_coupled_steps; icoupled++) {
            // block Gauss-seidel strategy
            ua = transfer.transferDisps(us);
            aero_solver.solve(ua);  // or iterate
            fa = aero_solver.getAeroLoads();
            fs = transfer.transferLoads(fa);
            struct_solver.solve(fs);
            us = struct_solver.getStructDisps();
        }
    }

   private:
    int num_coupled_steps;
    StructSolver struct_solver;
    AeroSolver aero_solver;
    Transfer transfer;

    int ns, na;
    Vec us, ua, fs, fa;
};