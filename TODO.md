# TODO
*Type Ctrl+Shift+V to view in VScode*
## <span style="color:#5bc0eb">Current Tasks</span>
Note, writeup what you're doing as you complete each major step in the overleaf.

### Current
- [ ] linear solver work on GPU
    * BSR works better with iterative solve, CSR better with direct solve
    * which linear solver is probably application dependent:
        * for coupled analysis & optimization : prob want direct CSR solve because of many solves (iterative BSR if possible but maybe not possible)
        * for regular struct opt with single linear solve : want iterative BSR solve is fastest (iterative solves fastest on single linear system)
    - [ ] add destructors for each object
    - [ ] add CSR direct solver (first task)
        - [ ] make CSR matrix and CSR data classes that inherit from BSR style with block_dim = 1
        - [ ] also need to convert elem_ind_map probably for CSR maybe
    - [ ] add BSR iterative solver (do we need preconditioner also)
        - [ ] how do we preconditioner the BSR on GPU? (ILU preconditioner) Or can we do that on CPU side with multi-procs? cause cusparse doesn't do ILU on GPU for BSR
    - [ ] finally : can we do BSR direct solver (talk with Kevin)

- [x] make a CPU linear solver for BSR first, and that way I can have additional debugging level in C++
- [ ] speedup BC kernel if needed?
- [x] verify linear solve of plate with sin-sin loads on 5 x 5 rect mesh
    - [x] it runs the solve now, but gives wrong answer
    - [x] verify same kmat (fixes with sorting elem_conn => better way to do this?)


### <span style="color:#fde74c">Basic Linear Solve with Shell Element on GPU</span>
- [ ] fully verify and optimize add residual on GPU
    - [x] demo add residual for shell element
    - [x] XDSM diagram of each method
    - [x] concept for optimal memory storage in main method
    - [x] update code with optimal memory storage in the main method
    - [x] verify element strain energy against TACS
    - [x] verify element energy derivs with complex step on CPU
    - [x] verify global residual for one element against TACS on CPU
    - [x] fix nan issue in global residual assembly
    - [x] verify CPU and GPU give same global residual
    ---- big checkpoint : working residual runs on GPU! matches TACS
    - [ ] significantly reduce register pressure by reusing more of the same arrays. Use references to the same array in memory.
    - [ ] make a chart showing the amount of temp memory stored at each point in the scripts
    - [ ] NVIDIA profiling
    - [ ] optimize interp tying strain method
    - [ ] optimize compute tying strain method
    - [ ] add each of these results to overleaf
    - [ ] add these to a new ppt
- [ ] fully verify and optimize add jacobian on GPU
    - [x] demo of running jacobian
    - [ ] XDSM diagram of each method
    - [ ] concept for optimal memory storage
    - [x] fix nans in jacobian
    - [x] verify element jac derivs with complex step
    - [x] verify element jacobian against TACS (why does it not match but res matches jac?)
    - [x] verify CPU vs GPU jacobian are equivalent
    ---- big checkpoint : working jacobian runs on GPU! matches TACS
    - [ ] NVIDIA profiling
    - [ ] add each of these results to overleaf
- [ ] linear solve on the GPU
    - [x] get linear solve to work with cuSparse
    - [x] add_bcs for vec.h
    - [x] add_bcs for bsrMat.h
    - [x] fix 0 soln in examples/basic_shell/solve_cusparse_gpu.cu
    - [x] Seiyon to help get ILU(k) fill pattern to work for more accurate cusparse solve 
    - [ ] verify small mesh linear solve against TACS

### More additions
- [ ] geometric nonlinear verification
    - [ ] add nonlinear tying strain hfwd, hrev code
    - [ ] verify nonlinear residual against TACS
    - [ ] verify nonlinear jacobian against TACS
- [ ] extend to nonlinear directors (and test jacobian vs res with CS)
- [ ] add blade stiffened constitutive physics? and associated ksfailure there..

### Performance Improvements
From Kevin:
- [ ] somtimes want to specify unusual launch parameters like 34 threads per block is optimal after checking on your device (not always multiple of 32)
- [ ] Do warp shuffle before adding into shared (adds among threads first with reduction) => this reduces number of atomic adds
- [ ] CUDA asynchronous shared memory load (look it up in performance guide)
Other:
- [ ] go through add residual, add jacobian of shell element => compute as many things on the fly as possible store minimum amount of data.
for example, compute transpose of matrix in some cases so columns available as rows now in consecutive memory. etc.
- [ ] show this approach of computing on the fly is X amount faster than doing it without that.. or show in ppt very well
- [ ] NVIDIA profiling of add residual, add jacobian kernels. Send results to Kevin, Ali, Dr. K
- [ ] profile on GPU best ways to interp tying strain (time that part only). Maybe make small script to test this out..
- [ ] compare computing then storing interpolation vecs N[i] in basis on GPU vs. procedurally getting each N(i) value in a loop using method call for that entry.

### <span style="color:#9bc53d">Tasks for Scitech</span>
- [ ] add thermal strains into the formulation as well
- [ ] get G matrix with third order directional derivative for stability
- [ ] get fourth-order stability matrix for post-buckling
- [ ] get adjoint-jacobian vector products for optimization
- [ ] demonstrate faster eigenvalue / linear solves of cylinders for thermal buckling presentation (Sean)
- [ ] do some asymptotic postbuckling analysis on the cylinders? See Bao's paper?

### Future Work
- [ ] preconditioners and algebraic multigrid for linear solver?
- [ ] matrix-free solve?

## <span style="color:#fe4a49">Completed tasks</span>
- [x] demo add residual for plane stress triangle element
- [x] demo add jacobian for plane stress triangle element   