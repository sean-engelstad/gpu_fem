# TODO

*Type Ctrl+Shift+V to view in VScode*

## Current
- [ ] put all of our code in namespace's
* By next week 2/10/2025
- [ ] implement reverse cuthill-mckee (RCM) from TACS_Utilities.cpp => use for better baseline symbolic factorization on GPU
- [ ] Q ordering for incomplete ILU solve (very fast iterative solve)
- [ ] implement iterative solve strategies with cusparse
- [ ] get setup on NAS to run GPU code and install nvcc locally on my laptop
- [ ] ssh into Meryem's desktop machine.. to run GPU stuff in the lab

By two weeks 2/17/2025
- [ ] put MELD on GPU

## 2/3/2025 meeting notes
Reordering -> Nested dissection –> METIS will be much more efficient
76% compute throughput because you are memory-bound
Speed of light not as important
Prioritize nonlinear solve over MDAO stuff before May
Linear solver is the bottle neck
Q-ordering on ILU -> GMRES -> Read the paper by Kevin [Node Numbering for Stabilizing Preconditioners Based on Incomplete LU Decomposition](https://arc.aiaa.org/doi/epdf/10.2514/6.2020-3022)
RCM Ordering, Cuthill Mckee ordering in TACS utilities cpp
Q ordering + ILU -> speed up? -> May help with nonlinear solves
Direct factorization acts as a fallback or basline
Look in TACS for METIX nested dissections TACS Assembler
Prioritize nonlinear structures solve over the optimization (adjoint, dRdx kernels, etc.) [Kevin]
Prioritize speeding up solver over the assembly
