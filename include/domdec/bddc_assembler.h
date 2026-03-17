#pragma once

#include <chrono>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "_fetidp.cuh"
#include "cuda_utils.h"
#include "domdec/fetidp_assembler.h"
#include "element/shell/_shell.cuh"
#include "linalg/bsr_data.h"
#include "multigrid/solvers/solve_utils.h"

template <typename T, class ShellAssembler_, template <typename> class Vec_,
          template <typename> class Mat_>
class BddcSolver : public FetidpSolver<T, ShellAssembler_, Vec_, Mat_> {
    // adapted from FETI-DP solver in same folder

   public:
    using ShellAssembler = ShellAssembler_;
    using Director = typename ShellAssembler::Director;
    using Basis = typename ShellAssembler::Basis;
    using Geo = typename Basis::Geo;
    using Phys = typename ShellAssembler::Phys;
    using Data = typename Phys::Data;
    using Quadrature = typename Basis::Quadrature;
    using FADType = typename A2D::ADScalar<T, 1>;
    using Vec = Vec_<T>;
    using Mat = Mat_<Vec_<T>>;
    using BsrMatType = BsrMat<DeviceVec<T>>;

    static constexpr int32_t nodes_per_elem = Basis::num_nodes;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;
    static constexpr int32_t xpts_per_elem = Geo::spatial_dim * nodes_per_elem;
    static constexpr int32_t dof_per_elem = vars_per_node * nodes_per_elem;
    static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

    BddcSolver() = default;

    BddcSolver(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
               ShellAssembler &assembler_, BsrMatType &kmat_, bool print_timing_ = false,
               bool warnings_ = true)
        : FetidpSolver<T, ShellAssembler_, Vec_, Mat_>(cublasHandle_, cusparseHandle_, assembler_,
                                                       kmat_, print_timing_) {
        warnings = warnings_;
    }

    ~BddcSolver() { this->clear_host_data(); }

    int getLambdaSize() const { return ngam * this->block_dim; }

    void setup_structured_subdomains(int nxe_, int nye_, int nxs_, int nys_,
                                     bool close_hoop = false) {
        // call base FETI-DP setup
        FetidpSolver<T, ShellAssembler_, Vec_, Mat_>::setup_structured_subdomains(nxe_, nye_, nxs_,
                                                                                  nys_, close_hoop);

        // TODO: BDDC unique maps/weights for edge averaging

        // build vectors of size gam
        n_edge = this->lam_nnodes;
        this->ngam = n_edge + this->Vc_nnodes;
        gam_nodes = new int[this->ngam];

        for (int i = 0; i < n_edge; i++) {
            gam_nodes[i] = this->lam_nodes[i];
        }
        for (int i = n_edge; i < ngam; i++) {
            gam_nodes[i] = this->Vc_nodes[i - n_edge];
        }
        // printf("gam_nodes (nE %d, nV %d): ", n_edge, this->Vc_nnodes);
        // printVec<int>(ngam, gam_nodes);

        this->temp_lam = Vec(this->lam_nnodes * this->block_dim);
        this->temp_lam2 = Vec(this->lam_nnodes * this->block_dim);
    }

    void update_after_assembly(DeviceVec<T> &vars) {
        // copy vars to d_vars (NL update)
        vars.copyValuesTo(this->d_vars);
        this->assemble_subdomains();
        this->subdomainIESolver->factor();
        this->subdomainISolver->factor();
        this->assemble_coarse_problem();
        this->coarseSolver->factor();
    }

    template <int elems_per_block = 8>
    void set_IEV_residual(T lambdaE, T lambdaI, DeviceVec<T> vars) {
        // res_IEV(u_IEV) = lambdaE * fext_IEV - lambdaI * fint_IEV

        // printf("set_IEV_residual\n");

        this->addVec_globalToIEV(this->d_xpts, this->d_IEV_xpts, 3, 1.0, 0.0);
        this->addVec_globalToIEV(vars, this->d_IEV_vars, this->block_dim, 1.0, 0.0);

        this->fint_IEV.zeroValues();

        dim3 block(num_quad_pts, elems_per_block);
        dim3 grid(this->num_elements);

        k_add_residual_fast<T, elems_per_block, ShellAssembler, Data, Vec_>
            <<<grid, block>>>(this->IEV_nnodes, this->num_elements, this->d_elem_components,
                              this->d_IEV_elem_conn, this->d_IEV_elem_conn, this->d_IEV_xpts,
                              this->d_IEV_vars, this->d_compData, this->fint_IEV);
        this->fint_IEV.apply_bcs(this->d_IEV_bcs);

        T a = -lambdaI;
        CHECK_CUBLAS(cublasDscal(this->cublasHandle, this->block_dim * this->IEV_nnodes, &a,
                                 this->fint_IEV.getPtr(), 1));

        this->fint_IEV.copyValuesTo(this->res_IEV);
        a = lambdaE;
        CHECK_CUBLAS(cublasDaxpy(this->cublasHandle, this->block_dim * this->IEV_nnodes, &a,
                                 this->fext_IEV.getPtr(), 1, this->res_IEV.getPtr(), 1));
    }

    void set_inner_solvers(BaseSolver *subdomainIESolver_, BaseSolver *subdomainISolver_,
                           BaseSolver *coarseSolver_, BaseSolver *subdomainIKrylov = nullptr) {
        // subdomainIKrylov matrix can be avoided only if use full fillin on K_II
        this->subdomainIESolver = subdomainIESolver_;
        this->subdomainISolver = subdomainISolver_;
        this->coarseSolver = coarseSolver_;
        this->subdomainIKrylov = subdomainIKrylov;
    }

    void get_lam_rhs(DeviceVec<T> &gam_rhs) {
        gam_rhs.zeroValues();
        this->res_IEV.copyValuesTo(this->f_IEV);
        this->addVecIEVtoIE(this->f_IEV, this->f_IE, 1.0, 0.0);
        this->addVecIEtoI(this->f_IE, this->f_I, 1.0, 0.0);
        // this part here must use full fillin / Krylov solve on K_II subdomain parallel matrix
        this->solveSubdomainIKrylov(this->f_I, this->u_I);
        this->addVecItoIE(this->u_I, this->u_IE, 1.0, 0.0);
        this->addVecIEtoIEV(this->u_IE, this->u_IEV, 1.0, 0.0);
        this->sparseMatVec(*this->kmat_IEV, this->u_IEV, -1.0, 1.0, this->f_IEV);
        this->addVecIEVtoGam(this->f_IEV, gam_rhs, 1.0, 0.0);

        // T *h_int_rhs = this->u_I.createHostVec().getPtr();
        // printf("h_u_I in gam rhs:\n");
        // for (int ilam = 0; ilam < this->I_nnodes; ilam++) {
        //     int iglob = this->I_nodes[ilam];
        //     printf("i_int %d, glob node %d: ", ilam, iglob);
        //     for (int idof = 2; idof < 5; idof++) {
        //         int lam_dof = this->block_dim * ilam + idof;
        //         printf("%.6e,", h_int_rhs[lam_dof]);
        //     }
        //     printf("\n");
        // }
    }

    void mat_vec(DeviceVec<T> &gam_in, DeviceVec<T> &gam_out) {
        gam_out.zeroValues();

        // const T *h_gam_in = gam_in.createHostVec().getPtr();
        // printf("h_gam_in_mat_vec:\n");
        // for (int ilam = 0; ilam < ngam; ilam++) {
        //     int iglob = gam_nodes[ilam];
        //     printf("igam %d, glob node %d: ", ilam, iglob);
        //     for (int idof = 2; idof < 5; idof++) {
        //         int lam_dof = this->block_dim * ilam + idof;
        //         printf("%.6e,", h_gam_in[lam_dof]);
        //     }
        //     printf("\n");
        // }

        this->addVecGamtoIEV(gam_in, this->u_IEV, 1.0, 0.0);
        this->sparseMatVec(*this->kmat_IEV, this->u_IEV, 1.0, 0.0, this->f_IEV);

        // solve rest of local Schur complement
        this->addVecIEVtoIE(this->f_IEV, this->f_IE, 1.0, 0.0);
        this->addVecIEtoI(this->f_IE, this->f_I, 1.0, 0.0);
        this->solveSubdomainI(this->f_I, this->u_I);

        this->addVecItoIE(this->u_I, this->u_IE, 1.0, 0.0);
        this->addVecIEtoIEV(this->u_IE, this->u_IEV, 1.0, 0.0);
        this->sparseMatVec(*this->kmat_IEV, this->u_IEV, -1.0, 1.0, this->f_IEV);
        this->addVecIEVtoGam(this->f_IEV, gam_out, 1.0, 0.0);

        // const T *h_gam_soln = gam_out.createHostVec().getPtr();
        // printf("h_gam_out_mat_vec:\n");
        // for (int ilam = 0; ilam < ngam; ilam++) {
        //     int iglob = gam_nodes[ilam];
        //     printf("igam %d, glob node %d: ", ilam, iglob);
        //     for (int idof = 2; idof < 5; idof++) {
        //         int lam_dof = this->block_dim * ilam + idof;
        //         printf("%.6e,", h_gam_soln[lam_dof]);
        //     }
        //     printf("\n");
        // }
    }

    bool solve(DeviceVec<T> gam_rhs, DeviceVec<T> gam, bool check_conv = false) {
        // does edge averaging (not vertex averaging since that's primal S_VV)
        const bool SCALED = true;

        // const T *h_gam_rhs = gam_rhs.createHostVec().getPtr();
        // printf("h_gam_rhs-pc:\n");
        // for (int ilam = 0; ilam < ngam; ilam++) {
        //     int iglob = gam_nodes[ilam];
        //     printf("igam %d, glob node %d: ", ilam, iglob);
        //     for (int idof = 2; idof < 5; idof++) {
        //         int lam_dof = this->block_dim * ilam + idof;
        //         printf("%.6e,", h_gam_rhs[lam_dof]);
        //     }
        //     printf("\n");
        // }

        // similar to FETI-DP mat_vec (flipped), but a bit different
        this->addVecGamtoIEV<SCALED>(gam_rhs, this->f_IEV, 1.0, 0.0);

        // debug check initial V_rhs
        // for vertices in rectangular part (will need to change this for wing case here)
        // TODO : change this part for wing case here..
        this->addVecIEVtoVc(this->f_IEV, this->f_V, 0.25, 0.0);

        // IE solve
        this->addVecIEVtoIE(this->f_IEV, this->f_IE, 1.0, 0.0);
        this->addVecIEtoIEV(this->f_IE, this->f_IEV, -1.0, 1.0);  // remove IE part
        this->zeroInteriorIE(this->f_IE);
        this->solveSubdomainIE(this->f_IE, this->u_IE);

        this->addVecIEtoIEV(this->u_IE, this->u_IEV, 1.0, 0.0);
        this->sparseMatVec(*this->kmat_IEV, this->u_IEV, -1.0, 0.0, this->f_IEV);

        // coarse solve
        this->addVecIEVtoVc(this->f_IEV, this->f_V, 1.0, 1.0);
        this->solveCoarse(this->f_V, this->u_V);

        // harmonic extension back to edge space
        this->addVecVctoIEV(this->u_V, this->temp_IEV, 1.0, 0.0);
        this->sparseMatVec(*this->kmat_IEV, this->temp_IEV, -1.0, 0.0, this->f_IEV);
        this->addVecIEVtoIE(this->f_IEV, this->f_IE, 1.0, 0.0);
        this->u_IE.zeroValues();

        this->solveSubdomainIE(this->f_IE, this->u_IE);
        this->addVecIEtoIEV(this->u_IE, this->u_IEV, 1.0, 1.0);

        // add u_V into u_IEV
        // TODO : generalize better than 0.25 here (for wing case)
        this->addVecVctoIEV(this->u_V, this->u_IEV, 0.25, 1.0);

        // now IEV to gam with averaging
        this->addVecIEVtoGam<SCALED>(this->u_IEV, gam, 1.0, 0.0);

        // const T *h_gam = gam.createHostVec().getPtr();
        // printf("h_gam-pc:\n");
        // for (int ilam = 0; ilam < ngam; ilam++) {
        //     int iglob = gam_nodes[ilam];
        //     printf("igam %d, glob node %d: ", ilam, iglob);
        //     for (int idof = 2; idof < 5; idof++) {
        //         int lam_dof = this->block_dim * ilam + idof;
        //         printf("%.6e,", h_gam[lam_dof]);
        //     }
        //     printf("\n");
        // }

        return false;  // fail = false
    }

    void get_global_soln(DeviceVec<T> &gam, DeviceVec<T> &soln) {
        // recover global solution from interface DOF
        soln.zeroValues();

        // const T *h_gam_soln = gam.createHostVec().getPtr();
        // printf("h_gam_soln:\n");
        // for (int ilam = 0; ilam < ngam; ilam++) {
        //     int iglob = gam_nodes[ilam];
        //     printf("igam %d, glob node %d: ", ilam, iglob);
        //     for (int idof = 2; idof < 5; idof++) {
        //         int lam_dof = this->block_dim * ilam + idof;
        //         printf("%.6e,", h_gam_soln[lam_dof]);
        //     }
        //     printf("\n");
        // }

        // set IE values from interface to IEV subdomains
        this->addVecGamtoIEV(gam, this->u_IEV, 1.0, 0.0);

        // first add the solved E and V DOF into IE and V parts (no scaling 1.0)
        // this->addVecGamtoIEV()

        this->addVecIEVtoVc(this->u_IEV, this->u_V, 0.25, 0.0);
        this->addVecIEVtoI(this->res_IEV, this->f_I, 1.0, 0.0);
        this->sparseMatVec(*this->kmat_IEV, this->u_IEV, -1.0, 0.0, this->f_IEV);
        this->addVecIEVtoI(this->f_IEV, this->f_I, 1.0, 1.0);
        // this part here must use full fillin / Krylov solve on K_II subdomain parallel matrix
        this->solveSubdomainIKrylov(this->f_I, this->u_I);
        this->addVecItoIE(this->u_I, this->u_IE, 1.0, 0.0);

        // add from gam the edges into u_IE
        this->addVecGamtoIEV(gam, this->u_IEV, 1.0, 0.0);
        this->addVecIEVtoIE(this->u_IEV, this->u_IE, 1.0, 1.0);

        this->addGlobalSoln(this->u_IE, this->u_V, soln);
    }

    BsrMat<DeviceVec<T>> *getKmatI() { return this->kmat_I; }
    int getInvars() { return this->I_nnodes * this->block_dim; }

   private:
    // additional utilities
    void zeroIEinIEV(DeviceVec<T> &vec_IEV) {
        this->addVecIEVtoIE(vec_IEV, this->temp_IE, 1.0, 0.0);
        this->addVecIEtoIEV(this->temp_IE, vec_IEV, -1.0, 1.0);
    }

    void zeroIinIEV(DeviceVec<T> &vec_IEV) {
        this->addVecIEVtoIE(vec_IEV, this->temp_IE, 1.0, 0.0);
        this->addVecIEtoI(this->temp_IE, this->temp_I, 1.0, 0.0);
        this->addVecIEtoI(this->temp_I, this->temp_IE, 1.0, 0.0);
        this->addVecIEtoIEV(this->temp_IE, vec_IEV, -1.0, 1.0);
    }

    void solveSubdomainIKrylov(Vec &rhs_in, Vec &sol_out) {
        auto _bsr_data = this->kmat_I->getBsrData();
        int *d_perm = _bsr_data.perm, *d_iperm = _bsr_data.iperm;
        rhs_in.copyValuesTo(this->rhs_I_perm);
        this->rhs_I_perm.permuteData(this->block_dim, d_iperm);

        if (subdomainIKrylov == nullptr) {
            if (warnings)
                printf(
                    "WARNING : must have full fillin if using non-Krylov solver for full K_II^-1 "
                    "solve\n\tIf want ILU(k) fillin, use set_inner_solvers to set subdomainIKrylov "
                    "solver.\n");
            this->subdomainISolver->solve(this->rhs_I_perm, this->sol_I_perm);
        } else {
            // do want to check convergence for this one
            subdomainIKrylov->solve(this->rhs_I_perm, this->sol_I_perm, true);
        }

        this->sol_I_perm.copyValuesTo(sol_out);
        sol_out.permuteData(this->block_dim, d_perm);
    }

    template <bool SCALED = false>
    void addVecIEVtoGam(const DeviceVec<T> &vec_IEV, DeviceVec<T> &vec_gam, T alpha, T beta) {
        // add from IEV to unique EV indices (lam plus V DOF)

        int gam_size = ngam * this->block_dim;
        CHECK_CUBLAS(cublasDscal(this->cublasHandle, gam_size, &beta, vec_gam.getPtr(), 1));

        // add IEV to lam first (edge DOF)
        this->addVecIEVtoIE(vec_IEV, this->temp_IE, alpha, 0.0);
        addVecIEtoGam(this->temp_IE, this->temp_lam, 1.0, 0.0);

        T a = SCALED ? 0.5 : 1.0;
        int edge_size = this->lam_nnodes * this->block_dim;
        CHECK_CUBLAS(cublasDaxpy(this->cublasHandle, edge_size, &a, this->temp_lam.getPtr(), 1,
                                 vec_gam.getPtr(), 1));

        // now add Vc part in next block in vector
        this->addVecIEVtoVc(vec_IEV, this->temp_V, alpha, 0.0);

        T *d_vec_gam = vec_gam.getPtr();
        a = 1.0;
        int V_size = this->Vc_nnodes * this->block_dim;
        CHECK_CUBLAS(cublasDaxpy(this->cublasHandle, V_size, &a, this->temp_V.getPtr(), 1,
                                 &d_vec_gam[edge_size], 1));
    }

    template <bool SCALED = false>
    void addVecGamtoIEV(const DeviceVec<T> &vec_gam, DeviceVec<T> &vec_IEV, T alpha, T beta) {
        int IEV_size = this->block_dim * this->IEV_nnodes;
        CHECK_CUBLAS(cublasDscal(this->cublasHandle, IEV_size, &beta, vec_IEV.getPtr(), 1));

        this->temp_lam.zeroValues();
        this->temp_V.zeroValues();

        T a = SCALED ? 0.5 : 1.0;
        a *= alpha;
        int edge_size = this->lam_nnodes * this->block_dim;
        CHECK_CUBLAS(cublasDaxpy(this->cublasHandle, edge_size, &a, vec_gam.getPtr(), 1,
                                 this->temp_lam.getPtr(), 1));

        addVecGamtoIE(this->temp_lam, this->temp_IE, 1.0, 0.0);
        this->addVecIEtoIEV(this->temp_IE, vec_IEV, 1.0, 0.0);

        a = alpha;
        int V_size = this->Vc_nnodes * this->block_dim;
        CHECK_CUBLAS(cublasDaxpy(this->cublasHandle, V_size, &a, &vec_gam.getPtr()[edge_size], 1,
                                 this->temp_V.getPtr(), 1));
        this->addVecVctoIEV(this->temp_V, vec_IEV, 1.0, 1.0);
    }

    void addVecGamtoIE(const DeviceVec<T> &gam, DeviceVec<T> &vec_IE, T a, T b) {
        // map(a * x) + b * y => y
        CHECK_CUBLAS(cublasDscal(this->cublasHandle, vec_IE.getSize(), &b, vec_IE.getPtr(), 1));
        int nvals = this->IE_nnodes * this->block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecGamtoIE<T><<<grid, block>>>(this->IE_nnodes, this->block_dim, this->d_IE_to_lam_map,
                                            gam.getPtr(), vec_IE.getPtr(), a);
    }

    void addVecIEtoGam(const DeviceVec<T> &vec_IE, DeviceVec<T> &gam, T a, T b) {
        // map(a * x) + b * y => y
        CHECK_CUBLAS(cublasDscal(this->cublasHandle, gam.getSize(), &b, gam.getPtr(), 1));
        int nvals = this->IE_nnodes * this->block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecIEtoGam<T><<<grid, block>>>(this->IE_nnodes, this->block_dim, this->d_IE_to_lam_map,
                                            vec_IE.getPtr(), gam.getPtr(), a);
    }

   private:
    bool warnings;
    int ngam, n_edge;
    int gam_offset;
    DeviceVec<T> temp_lam, temp_lam2;
    int *gam_nodes;
    BaseSolver *subdomainIKrylov;
};