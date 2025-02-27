
## 3/3/2025 meeting notes (TBD)


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