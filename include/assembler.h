#pragma once

#include <cstring>

#include "base/elem_group.h"
#include "chrono"
#include "cuda_utils.h"
#include "mesh/TACSMeshLoader.h"

// linear algebra formats
#include "linalg/bsr_utils.h"
#include "linalg/vec.h"

template <typename T_, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat_>
class ElementAssembler
{
public:
    using T = T_;
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Data = typename Phys::Data;
    using Mat = Mat_<Vec<T>>;
    static constexpr int32_t geo_nodes_per_elem = Geo::num_nodes;
    static constexpr int32_t vars_nodes_per_elem = Basis::num_nodes;
    static constexpr int32_t spatial_dim = Geo::spatial_dim;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;

    // dummy constructor for random points (another one will be made for actual
    // connectivity)
    ElementAssembler(int32_t num_geo_nodes_, int32_t num_vars_nodes_,
                     int32_t num_elements_, HostVec<int32_t> &geo_conn,
                     HostVec<int32_t> &vars_conn, HostVec<T> &xpts,
                     HostVec<int> &bcs, HostVec<Data> &physData)
        : num_geo_nodes(num_geo_nodes_), num_vars_nodes(num_vars_nodes_),
          num_elements(num_elements_)
    {

        // keeping inputs as HostVec even if running on device eventually here
        // std::unique, std::sort, and std::vector not directly supported on GPU
        // there are some options like thrust::sort, thrust::device_vector,
        // thrust::unique but I would also need to launch kernel for BSR..
        // should be cheap enough to just do on host now and then
        // createDeviceVec here

        int32_t num_vars = get_num_vars();
        this->vars = Vec<T>(num_vars);

        // on host (TODO : if need to deep copy entries to device?)
        // TODO : should probably do factorization explicitly instead of
        // implicitly upon construction
        bsr_data = BsrData(num_elements, num_vars_nodes, Basis::num_nodes,
                           Phys::vars_per_node, vars_conn.getPtr());

#ifdef USE_GPU

        // convert everything to device vecs
        this->geo_conn = geo_conn.createDeviceVec();
        this->vars_conn = vars_conn.createDeviceVec();
        this->xpts = xpts.createDeviceVec();
        this->bcs = bcs.createDeviceVec();
        this->physData = physData.createDeviceVec(false);

#else // not USE_GPU

        // on host just copy normally
        this->geo_conn = geo_conn;
        this->vars_conn = vars_conn;
        this->xpts = xpts;
        this->bcs = bcs;
        this->physData = physData;

#endif // end of USE_GPU or not USE_GPU check
    };

    BsrData getBsrData() { return bsr_data; }
    void symbolic_factorization(double fillin = 100.0, bool print = false)
    {
        bsr_data.symbolic_factorization(fillin, print);
#ifdef USE_GPU
        this->bsr_data = bsr_data.createDeviceBsrData();
#endif
    }
    void bsrDataToDevice(double fillin = 100.0, bool print = false)
    {
        // move bsr data to device for single Kelem (for debugging)
#ifdef USE_GPU
        this->bsr_data = bsr_data.createDeviceBsrData();
#endif
    }

    // main way to construct an ElementAssembler from a BDF file
    // as long as that BDF file has only one element type (TODO on multiple
    // element types later)
    static ElementAssembler createFromBDF(TACSMeshLoader<T> &mesh_loader,
                                          Data single_data)
    {
        int vars_per_node = Phys::vars_per_node; // input

        int num_nodes, num_elements, num_bcs;
        int *elem_conn, *bcs;
        double *xpts;

        mesh_loader.getAssemblerCreatorData(vars_per_node, num_nodes,
                                            num_elements, num_bcs, elem_conn,
                                            bcs, xpts);

        // make HostVec objects here for Assembler
        HostVec<int> elem_conn_vec(vars_nodes_per_elem * num_elements,
                                   elem_conn);
        HostVec<int> bcs_vec(num_bcs, bcs);
        HostVec<T> xpts_vec(spatial_dim * num_nodes, xpts);
        HostVec<Data> physData_vec(num_elements, single_data);

        // call base constructor
        return ElementAssembler(num_nodes, num_nodes, num_elements,
                                elem_conn_vec, elem_conn_vec, xpts_vec, bcs_vec,
                                physData_vec);
    }

    int get_num_xpts() { return num_geo_nodes * spatial_dim; }
    int get_num_vars() { return num_vars_nodes * vars_per_node; }
    __HOST__ void apply_bcs(Vec<T> &vec, bool can_print = false)
    {
        if (can_print)
        {
            printf("apply bcs to vector\n");
        }
        auto start = std::chrono::high_resolution_clock::now();

        vec.apply_bcs(bcs);

        // print timing data
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        double dt = duration.count() / 1e6;
        if (can_print)
        {
            printf("\tfinished apply bcs vec in %.4e seconds\n",
                   dt);
        }
    }
    void apply_bcs(Mat &mat, bool can_print = false)
    {
        if (can_print)
        {
            printf("apply bcs to matrix\n");
        }
        auto start = std::chrono::high_resolution_clock::now();

        mat.apply_bcs(bcs);

        // print timing data
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        double dt = duration.count() / 1e6;
        if (can_print)
        {
            printf("\tfinished apply bcs matrix in %.4e sec\n",
                   dt);
        }
    }

    HostVec<T> createVarsHostVec(T *data = nullptr, bool randomize = false)
    {
        HostVec<T> h_vec;
        if (data == nullptr)
        {
            h_vec = HostVec<T>(get_num_vars());
        }
        else
        {
            h_vec = HostVec<T>(get_num_vars(), data);
        }
        if (randomize)
        {
            h_vec.randomize();
        }
        return h_vec;
    }

#ifdef USE_GPU
    DeviceVec<T> createVarsVec(T *data = nullptr, bool randomize = false,
                               bool can_print = false)
    {
        if (can_print)
        {
            printf("begin create vars host vec\n");
        }
        auto h_vec = createVarsHostVec(data, randomize);
        if (can_print)
        {
            printf("inner checkpt 2\n");
        }
        auto d_vec = h_vec.createDeviceVec(true, can_print);
        if (can_print)
        {
            printf("inner checkpt 3\n");
        }
        return d_vec;
    }
#else
    HostVec<T> createVarsVec(T *data = nullptr, bool randomize = false)
    {
        return createVarsHostVec(data, randomize);
    }
#endif

    void set_variables(Vec<T> &newVars)
    {
        // vars is either device array on GPU or a host array if not USE_GPU
        // should we not do deep copy here?
        this->vars.setData(newVars.getPtr(), bsr_data.perm, bsr_data.block_dim);
    }

    //  template <class ExecParameters>
    void add_energy(T &Uenergy, bool can_print = false)
    {
        auto start = std::chrono::high_resolution_clock::now();
        if (can_print)
        {
            printf("begin add_energy\n");
        }

// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU
        dim3 block = ElemGroup::energy_block;
        int nblocks = (num_elements + block.x - 1) / block.x;
        dim3 grid(nblocks);
        constexpr int32_t elems_per_block = ElemGroup::res_block.x;

        add_energy_gpu<T, ElemGroup, Data, elems_per_block><<<grid, block>>>(
            num_elements, geo_conn, vars_conn, xpts, physData, Uenergy);

        CHECK_CUDA(cudaDeviceSynchronize());
#else  // USE_GPU
        ElemGroup::template add_energy_cpu<Data>(
            num_elements, geo_conn, vars_conn, xpts, vars, physData, Uenergy);
#endif // USE_GPU

        // print timing data
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        double dt = duration.count() / 1e6;
        if (can_print)
        {
            printf("\tfinished assembly in %.4e sec\n",
                   dt);
        }
    };

    //  template <class ExecParameters>
    void add_residual(Vec<T> &res, bool can_print = false)
    {
        auto start = std::chrono::high_resolution_clock::now();
        if (can_print)
        {
            printf("begin add_residual\n");
        }

// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU
        dim3 block = ElemGroup::res_block;
        int nblocks = (num_elements + block.x - 1) / block.x;
        dim3 grid(nblocks);
        constexpr int32_t elems_per_block = ElemGroup::res_block.x;

        add_residual_gpu<T, ElemGroup, Data, elems_per_block, Vec>
            <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars,
                              bcs, physData, bsr_data.perm, res);

        CHECK_CUDA(cudaDeviceSynchronize());
#else  // USE_GPU
        ElemGroup::template add_residual_cpu<Data, Vec>(
            num_elements, geo_conn, vars_conn, xpts, vars, physData, res);
#endif // USE_GPU

        // print timing data
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        double dt = duration.count() / 1e6;
        if (can_print)
        {
            printf("\tfinished assembly in %.4e\n",
                   dt);
        }
    };

    void compute_stresses(DeviceVec<T> &stresses, bool can_print = false)
    {
        auto start = std::chrono::high_resolution_clock::now();
        if (can_print)
        {
            printf("begin add_residual\n");
        }

// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU
        dim3 block = ElemGroup::stress_block;
        int nblocks = (num_elements + block.x - 1) / block.x;
        dim3 grid(nblocks);
        constexpr int32_t elems_per_block = ElemGroup::res_block.x;

        auto stress_cts = DeviceVec<int>(num_vars_nodes);

        compute_stresses_kernel<T, ElemGroup, Data, elems_per_block, Vec>
            <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars,
                              bcs, physData, stresses, stress_cts);

        // normalize the stresses and compute KS stress
        dim3 block2(32);
        dim3 grid2(num_vars_nodes);

        normalize_stresses_kernel<T, ElemGroup><<<grid2, block2>>>(stresses, stress_cts);

        CHECK_CUDA(cudaDeviceSynchronize());
#else  // USE_GPU
       // ElemGroup::template add_residual_cpu<Data, Vec>(
       //     num_elements, geo_conn, vars_conn, xpts, vars, physData, res);
#endif // USE_GPU

        // print timing data
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        double dt = duration.count() / 1e6;
        if (can_print)
        {
            printf("\tfinished compute stresses in %.4e\n",
                   dt);
        }
    };

    void evalFunctions(std::vector<Function> funcs)
    {
        for (func : funcs)
        {
            if (func type == mass)
            {
                _compute_mass(...)
            }
            else if (func type == ksfailure)
            {
                _compute_ks_failure(...)
            }
        }
    }

    void evalFunctionSens(std::vector<Function> funcs)
    {
        for (func : funcs)
        {
            if (func type == mass)
            {
                _compute_mass(...)
            }
            else if (func type == ksfailure)
            {
                _compute_ks_failure(...)
            }
        }
    }

    T _compute_ks_failure(T rho_KS, bool can_print = false)
    {

        auto start = std::chrono::high_resolution_clock::now();
        if (can_print)
        {
            printf("begin compute ks stress\n");
        }

#ifdef USE_GPU

        // compute nodal averaged stresses
        DeviceVec<T> stresses(num_vars_nodes * vars_per_node);
        compute_stresses(stresses, can_print);

        // compute ks stresses
        dim3 block(32);
        int nblocks = (num_vars_nodes + block.x - 1) / block.x;
        dim3 grid(nblocks);
        T sum_exp_fail;

        compute_ks_failure<T, Data>(stresses, rho_KS, &sum_exp_fail);

        T ks_failure = 1.0 / rho_KS * log(sum_exp_fail);

        return ks_failure;

#endif

        // print timing data
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        double dt = duration.count() / 1e6;
        if (can_print)
        {
            printf("\tfinished ks stresses in %.4e\n",
                   dt);
        }
    };

    //  template <class ExecParameters>
    void add_jacobian(Vec<T> &res, Mat &mat,
                      bool can_print = false)
    { // TODO : make this Vec here..
        auto start = std::chrono::high_resolution_clock::now();
        if (can_print)
        {
            printf("begin add_jacobian\n");
        }

// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU

        dim3 block = ElemGroup::jac_block;
        int nblocks = (num_elements + block.x - 1) / block.x;
        dim3 grid(nblocks);
        constexpr int32_t elems_per_block = ElemGroup::jac_block.x;

        add_jacobian_gpu<T, ElemGroup, Data, elems_per_block, Vec, Mat>
            <<<grid, block>>>(num_vars_nodes, num_elements, geo_conn, vars_conn,
                              xpts, vars, physData, res, mat);

        CHECK_CUDA(cudaDeviceSynchronize());

#else // CPU data
      // maybe a way to call add_residual_kernel as same method on CPU
      // with elems_per_block = 1
        ElemGroup::template add_jacobian_cpu<Data, Vec, Mat>(
            num_vars_nodes, num_elements, geo_conn, vars_conn, xpts, vars,
            physData, res, mat);
#endif

        // print timing data
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        double dt = duration.count() / 1e6;
        if (can_print)
        {
            printf("\tfinished assembly in %.4e sec\n",
                   dt);
        }
    };

    Vec<T> getXpts() { return xpts; }
    Vec<int> getConn() { return vars_conn; }
    int get_num_nodes() { return num_vars_nodes; }
    int get_num_elements() { return num_elements; }

private:
    int32_t num_geo_nodes;
    int32_t num_vars_nodes;
    int32_t num_elements; // Number of elements of this type

    Vec<int32_t> geo_conn, vars_conn;
    Vec<int> bcs;
    Vec<T> xpts, vars;
    Vec<Data> physData;
    BsrData bsr_data;
};