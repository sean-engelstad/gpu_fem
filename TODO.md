# TODO
*Type Ctrl+Shift+V to view in VScode*
## <span style="color:#5bc0eb">Current Tasks</span>
Note, writeup what you're doing as you complete each major step in the overleaf.

### <span style="color:#fde74c">Basic Linear Solve with Shell Element on GPU</span>
- [ ] fully verify and optimize add residual on GPU
    - [x] add_residual : demo add residual for shell element
    - [x] add residual : XDSM diagram of each method
    - [x] add residual : concept for optimal memory storage in main method
    - [x] add residual : update code with optimal memory storage in the main method
    - [ ] add residual : make a chart showing the amount of temp memory stored at each point in the scripts
    - [x] add residual : verify element strain energy against TACS
    - [x] add residual : verify element energy derivs with complex step on CPU
    - [x] add residual : verify global residual for one element against TACS on CPU
    - [x] add residual : fix nan issue in global residual assembly
    - [ ] add residual : verify CPU and GPU give same global residual
    - [ ] add residual : significantly reduce register pressure by reusing more of the same arrays. Use references to the same array in memory.
    - [ ] add residual : check global residual on larger mesh..
    - [ ] add residual : NVIDIA profiling
    - [ ] add residual : optimize interp tying strain method
    - [ ] add residual : optimize compute tying strain method
    - [ ] add residual : add each of these results to overleaf
    - [ ] add residual : add these to a new ppt
- [ ] fully verify and optimize add jacobian on GPU
    - [ ] add jacobian : demo of running jacobian
    - [ ] add jacobian : XDSM diagram of each method
    - [ ] add jacobian : concept for optimal memory storage
    - [x] add jacobian : fix nans in jacobian
    - [ ] add jacobian : make methods more general for nonlinear strain, director for pvalue, hvalue, etc.
    - [ ] add jacobian : verify element res derivs with complex step
    - [ ] add jacobian : verify element jacobian against TACS (one col at a time?)
    - [ ] add jacobian : verify global jacobian against TACS
    - [ ] add jacobian : NVIDIA profiling
    - [ ] add jacobian : look into shared memory compile issue with profiling?
    - [ ] add jacobian : add each of these results to overleaf
- [ ] linear solve on the GPU
    - [ ] get sparse element storage fill pattern from Aaron => need to 
    - [ ] get linear solver to work with cuSparse 

### Structural Optimizations in GPU_FEM
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
- [ ] generalize some methods for nonlinear tying strain, director, etc.
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