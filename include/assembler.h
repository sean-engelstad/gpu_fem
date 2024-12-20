#pragma once

#include <cstring>

#include "base/elem_group.h"
#include "cuda_utils.h"

// linear algebra formats
#include "linalg/bsr_utils.h"
#include "linalg/vec.h"

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat_>
class ElementAssembler {
  public:
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
          num_elements(num_elements_) {

        // keeping inputs as HostVec even if running on device eventually here
        // std::unique, std::sort, and std::vector not directly supported on GPU
        // there are some options like thrust::sort, thrust::device_vector,
        // thrust::unique but I would also need to launch kernel for BSR..
        // should be cheap enough to just do on host now and then
        // createDeviceVec here

        int32_t num_vars = get_num_vars();
        this->vars = Vec<T>(num_vars);

        // on host (TODO : if need to deep copy entries to device?)
        bsr_data = BsrData(num_elements, num_vars_nodes, Basis::num_nodes,
                           Phys::vars_per_node, vars_conn.getPtr());

#ifdef USE_GPU

        // convert everything to device vecs
        this->geo_conn = geo_conn.createDeviceVec();
        this->vars_conn = vars_conn.createDeviceVec();
        this->xpts = xpts.createDeviceVec();
        this->bcs = bcs.createDeviceVec();
        this->physData = physData.createDeviceVec(false);
        this->bsr_data = bsr_data.createDeviceBsrData();

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

    int get_num_xpts() { return num_geo_nodes * spatial_dim; }
    int get_num_vars() { return num_vars_nodes * vars_per_node; }
    __HOST__ void apply_bcs(Vec<T> &vec) {
        printf("here");
        vec.apply_bcs(bcs);
    }
    // void apply_bcs(Mat &mat) { mat.apply_bcs(bcs); }

    void set_variables(Vec<T> &vars) {
        // vars is either device array on GPU or a host array if not USE_GPU
        // should we not do deep copy here?
#ifdef USE_GPU
        cudaMemcpy(this->vars.getPtr(), vars.getPtr(),
                   this->vars.getSize() * sizeof(T), cudaMemcpyDeviceToDevice);
#else
        memcpy(this->vars.getPtr(), vars.getPtr(),
               this->vars.getSize() * sizeof(T));
#endif
    }

    //  template <class ExecParameters>
    void add_energy(T &Uenergy) {
// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU
        dim3 block = ElemGroup::energy_block;
        int nblocks = (num_elements + block.x - 1) / block.x;
        dim3 grid(nblocks);
        constexpr int32_t elems_per_block = ElemGroup::res_block.x;

        add_energy_gpu<T, ElemGroup, Data, elems_per_block><<<grid, block>>>(
            num_elements, geo_conn, vars_conn, xpts, physData, Uenergy);

        gpuErrchk(cudaDeviceSynchronize());
#else  // USE_GPU
        ElemGroup::template add_energy_cpu<Data>(
            num_elements, geo_conn, vars_conn, xpts, vars, physData, Uenergy);
#endif // USE_GPU
    };

    //  template <class ExecParameters>
    void add_residual(Vec<T> &res) {
// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU
        dim3 block = ElemGroup::res_block;
        int nblocks = (num_elements + block.x - 1) / block.x;
        dim3 grid(nblocks);
        constexpr int32_t elems_per_block = ElemGroup::res_block.x;

        add_residual_gpu<T, ElemGroup, Data, elems_per_block, Vec>
            <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars,
                              bcs, physData, res);

        gpuErrchk(cudaDeviceSynchronize());
#else  // USE_GPU
        ElemGroup::template add_residual_cpu<Data, Vec>(
            num_elements, geo_conn, vars_conn, xpts, vars, physData, res);
#endif // USE_GPU
    };

    //  template <class ExecParameters>
    void add_jacobian(Vec<T> &res, Mat &mat) { // TODO : make this Vec here..
// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU

        dim3 block = ElemGroup::jac_block;
        int nblocks = (num_elements + block.x - 1) / block.x;
        dim3 grid(nblocks);
        constexpr int32_t elems_per_block = ElemGroup::jac_block.x;

        add_jacobian_gpu<T, ElemGroup, Data, elems_per_block>
            <<<grid, block>>>(num_vars_nodes, num_elements, geo_conn, vars_conn,
                              xpts, vars, physData, res, mat);

        gpuErrchk(cudaDeviceSynchronize());

#else // CPU data
      // maybe a way to call add_residual_kernel as same method on CPU
      // with elems_per_block = 1
        ElemGroup::template add_jacobian_cpu<Data, Vec, Mat>(
            num_vars_nodes, num_elements, geo_conn, vars_conn, xpts, vars,
            physData, res, mat);
#endif
    };

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