## Multi-GPU Additive Schwarz PCG Demo for Shell Element Cylinder

* MITC4 shells
* Element-subdomain additive schwarz smoothing preconditioner
* implemented on multi-GPU format with single CPU proc
* note for very high DOF ASW-PCG doesn't converge (need GMG-ASW to fully converge) and just caps out at 500 Krylov iterations, but still can compare multi-GPU speedups

Current speedups for cylinder problem (of 4 A100 GPUs to 1 A100 GPU):
@SR = 10
NXE=512, NDOF=1.56M DOF, SPEEDUP=1.5x, 4 A100 GPU runtime = 
NXE=1024, NDOF=6.25M DOF, SPEEDUP=2.8x, 4 A100 GPU runtime = 4.24 sec
NXE=1500, NDOF=13.5M DOF, SPEEDUP=3.2x, 4 A100 GPU runtime = 7.74 sec

Memory limits (approx):
1 A100 GPU:  ~13.5M DOF
4 A100 GPUs: ~50M DOF

50M DOF runtime on 4 A100 GPUs = ~27 seconds

## Multi-GPU GMG-ASW K-cycle(PCG) for Shell Element Cylinder

* now run with full multigrid, not just ASW smoother on fine grid.

Current speedups (up to 4x) at each mesh level, a bit reduced from just ASW
cause each mesh level has a bit less speedup.


@SR = 10 (so can compare with ASW-PCG above)
NXE=1024, NDOF=6.25M DOF, SPEEDUP=1.61, 4 A100 GPU runtime = 0.88 sec
NXE=2048, NDOF=25.2M DOF, SPEEDUP=2.46, 4 A100 GPU runtime = 2.03 sec


@SR = 100
NXE=1024, NDOF=6.25M DOF, SPEEDUP=
NXE=2048, NDOF=25.2M DOF, SPEEDUP=2.98, 4 A100 GPU runtime = 4.98 sec

Probably at NXE=4096 (though that is too many DOF for 4 GPUs) to get full 4x speedup from 4 GPUs.