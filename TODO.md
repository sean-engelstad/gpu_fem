# TODO
*Type Ctrl+Shift+V to view in VScode*
## <span style="color:#5bc0eb">Current Tasks</span>
Note, writeup what you're doing as you complete each major step in the overleaf.

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
    - [ ] use Ali's methods for kernel global to shared mem transfer..
    - [ ] get sparse element storage fill pattern from Aaron => need to 
    - [ ] get linear solver to work with cuSparse 
    - [ ] verify small mesh linear solve against TACS

### More additions
- [ ] geometric nonlinear verification
    - [ ] add nonlinear tying strain hfwd, hrev code
    - [ ] verify nonlinear residual against TACS
    - [ ] verify nonlinear jacobian against TACS
- [ ] extend to nonlinear directors (and test jacobian vs res with CS)
- [ ] add blade stiffened constitutive physics? and associated ksfailure there..

### Performance Improvements
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