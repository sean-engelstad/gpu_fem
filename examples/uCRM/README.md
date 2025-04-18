# uCRM demonstration case

* goal is to demonstrate the uCRM wing for nonlinear static analyses, optimizations and coupled aeroelastic analysis
* NOTE : current issue is that direct solver can't achieve low ||K * u - F|| residual right now => suspect there is some ill-conditioning of the BCs which requires the use of an iterative solver
    * also the failure to converge the residual = f_int(u) - F = 0 for even the linear case to zero, means the current nonlinear static analysis with the direct solve doesn't converge either => need iterative solver.

## TODO:

- [ ] get iterative solver working and switch to that => will handle the ill-conditioning issue of the BCs
- [ ] fix MELD nans
- [ ] demo nonlinear static analysis
- [ ] demo coupled analysis, using fixed aero loads