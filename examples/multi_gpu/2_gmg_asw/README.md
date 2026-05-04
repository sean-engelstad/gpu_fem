# Multi-GPU Additive Schwarz PCG Demo for Shell Element Cylinder

* MITC4 shells
* Element-subdomain additive schwarz smoothing preconditioner
* implemented on multi-GPU format with single CPU proc
* note for very high DOF ASW-PCG doesn't converge (need GMG-ASW to fully converge) and just caps out at 500 Krylov iterations, but still can compare multi-GPU speedups

Current speedups for cylinder problem (of 4 A100 GPUs to 1 A100 GPU):
NXE=512, NDOF=1.56M DOF, SPEEDUP=1.5x, 4 A100 GPU runtime = 
NXE=1024, NDOF=6.25M DOF, SPEEDUP=2.8x, 4 A100 GPU runtime = 4.24 sec
NXE=1500, NDOF=13.5M DOF, SPEEDUP=3.2x, 4 A100 GPU runtime = 7.74 sec

Memory limits (approx):
1 A100 GPU:  ~13.5M DOF
4 A100 GPUs: ~50M DOF

50M DOF runtime on 4 A100 GPUs = ~27 seconds