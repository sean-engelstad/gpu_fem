# Discussion of Timing / Performance Profiling for FETI-DP on GPU

* NOTE : BDDC in other folder is actually more convenient for nonlinear and linear problems than
FETI-DP. Both solve in same # of iterations, but FETI-DP tracks the dual residual norm, while
BDDC tracks a displacement norm (kind of like right vs left preconditioning).
Thus, BDDC often ends up being more accurate to true residual (whereas FETI-DP requires a bit 
deeper PCG rtol to get same solution accuracy as BDDC).


## Single-GPU Profiling

* tried ILU(k) fillin of IE, I, S_VV matrices. Can solve with ILU(2) or ILU(3)
which lowers factor times considerably (coarse factor is very expensive)
* fill ILU(k) needs to be high enough depending on shell thickness, ILU(2) or ILU(3),
where MULTI_SMOOTH option didn't really help the performance.. makes it slower than not doing it. So turn that off, tried diff # smoothing steps + omega values.
* assembly, IEV-assembly and PCG-solve runtimes shown for several cases below

## V1-cylinder, single-GPU
* 256^2 elems, 396K DOF

update_after_assembly timing breakdown:
  assemble_subdomains   : 9.312310e-02 s
  factor IE             : 2.242282e-02 s
  factor I              : 6.143682e-03 s
  assemble coarse       : 1.872734e-01 s
    copyKmat_IEVtoSvv: 0.032608 ms
    computeSvvInverseTerm: 187.126785 ms
  factor coarse         : 1.025379e-02 s
  total                 : 3.192200e-01 s
  tracked subtotal      : 3.192168e-01 s
  untracked overhead    : 3.156000e-06 s

  MF-PCG init_resid = 1.40069639e-03
MF-PCG [0] = 6.66552801e-04
MF-PCG [5] = 3.84169264e-05
MF-PCG [10] = 1.98285417e-06
MF-PCG [15] = 1.18345592e-07
MF-PCG [20] = 4.93356599e-09
MF-PCG converged in 23 iterations to 1.260460223e-09 resid
debug: MF-PCG true residual 1.26046023e-09, x norm 2.48559761e+01

FETI-DP solve summary:
  final lambda residual : 1.26046022e-09

Timing breakdown:
  assembly              : 1.0094e-01 s
  IEV assembly+factor   : 3.1581e-01 s
  PCG solve             : 4.7412e-01 s
  --------------------------------
  total setup + solve   : 7.8993e-01 s

FETI-DP memory breakdown:
  kmat                 : nnzb = 590592, mem = 162.2109 MB
  IEV                  : nnzb = 692224, mem = 190.1250 MB
  IE                   : nnzb = 808704, mem = 222.1172 MB
    nofill             : 579328
    fill added         : 229376
    fill ratio         : 1.3959
  I                    : nnzb = 213376, mem = 58.6055 MB
    nofill             : 204672
    fill added         : 8704
    fill ratio         : 1.0425
  coarse S_VV          : nnzb = 110208, mem = 30.2695 MB
    nofill             : 35904
    fill added         : 74304
    fill ratio         : 3.0695
  --------------------------------
  total stored nnzb    : 3223808
  total memory         : 885.4453 MB

* compared to 256^2 direct-solve (at thick = 1e-3)
    * FETI-DP 0.789 sec, Direct-LU 3.121 sec
    * FETI-DP 885 MB, Direct-LU 1.55e3 MB
    * multigrid takes 7 or 8 seconds (we're 10x faster than that), cause breaks down at thin shell
    * thus: FETI-DP at this mid problem size has:
        3.9x speedup and 1.75x less memory

* expect to improve the time for IEV-assembly+factor (coarse Svv term)
* also FETI-DP may have less fillin compared to direct on higher DOF problems also.
    where Direct-LU solvers blow up in memory usage (over 30x fillins)
    also may have better mem comparison for wing too.
* and may be more friendly on multiple GPUs
* also note this is just C1-coarsening, vertex-based FETI-DP, I actually think this may be slightly faster than other methods (slightly more Krylov steps, but coarse problem is less fillin and less DOF than if extra constraints added). TBD or we'll see.

* IEV assembly+factor is actually quite slow, expect to improve that 
    by maybe speeding up the coarse assembly step of computing the SVV inverse term
* it does 24 triangular solves currently to compute coarse Svv-inverse-term

* performance of SvvInvereTerm step (pretty expensive, like 20-30% of runtime)
  S_{VV} = A_{VV} - A_{V,IE} * A_{IE,IE}^{-1} * A_{IE,V} 
  computes the second term which involves 24 triangular solves A_{IE,IE}^{-1}

  Timing results for this method:
  computeSvvInverseTerm: 187.531265 ms
          2 sparseMatVec total : 53.440254 ms
          solveSubdomainIE total: 130.096008 ms
          other total          : 3.994995 ms
          across 24 cols or steps here
  * basically 24 triang solves with subdomainIE solver is quite expensive (about 1/2 cost of PCG iterations or something)?
  * would prefer to do this mat-matinv-mat calculation at matrix level / more efficiently somehow to improve performance (gain like 30% back or something)

* Idk.. if can speedup Svv_InverseTerm more.. batched RHS triangular solves
  is not available for BSR matrices in CuSparse and I don't have BSR routines 
  to do triang-solve on matrix either.. but maybe still possible?
* Look at math and see if possible.. but not really sure..
* still it's less cost than the full PCG typically.. so maybe ok..
* just do multiple GPUs to further improve the performance

## FETI-DP nonlinear solve

## Try FETI-DP method on NAS A100 GPU to see speedup to direct-solve there..

