# <span style="color:#ffffff">TODO</span>

## <span style="color:#5bc0eb">Current Tasks</span>

### <span style="color:#fde74c">Basic Linear Solve on GPU</span>
- [ ] make a diagram showing when each variable is used at each call of add residual. Show where we can make scope blocks to store least amount of data possible.
Some kind of Gantt chart of something.
- [ ] demo add residual for shell element
- [ ] demo add jacobian for shell element
- [ ] verify strain energy for shell element against TACS
- [ ] verify add residual for shell element against complex-step
- [ ] verify add jacobian for shell element against complex-step or finite difference
- [ ] get linear solver to work with cuSparse 

### Performance Improvements
- [ ] go through add residual, add jacobian of shell element => compute as many things on the fly as possible store minimum amount of data.
for example, compute transpose of matrix in some cases so columns available as rows now in consecutive memory. etc.
- [ ] show this approach of computing on the fly is X amount faster than doing it without that.. or show in ppt very well
- [ ] NVIDIA profiling of add residual, add jacobian kernels. Send results to Kevin, Ali, Dr. K
- [ ] profile on GPU best ways to interp tying strain (time that part only). Maybe make small script to test this out..
- [ ] compare computing then storing interpolation vecs N[i] in basis on GPU vs. procedurally getting each N(i) value in a loop using method call for that entry.

### <span style="color:#9bc53d">Tasks for Scitech</span>
- [] get G matrix with third order directional derivative for stability
- [] get fourth-order stability matrix for post-buckling
- [] get adjoint-jacobian vector products for optimization
- [] demonstrate faster eigenvalue / linear solves of cylinders for thermal buckling presentation (Sean)


## <span style="color:#fe4a49">Completed tasks</span>
- [x] demo add residual for plane stress triangle element
- [x] demo add jacobian for plane stress triangle element