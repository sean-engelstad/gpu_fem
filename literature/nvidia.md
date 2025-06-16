# NVIDIA tips

* how to use nsight compute:
## 3.1 - nsight compute for double precision (CUDA profiling)

To use nsight compute profiler, you do the following
```
ncu --set full --export  report_name.ncu-rep ./exec_name.out
     (sometimes you need sudo and the full path to ncu executable such as sudo /usr/local/cuda-12.6/bin/ncu above)
aka:
   sudo /usr/local/cuda-12.6/bin/ncu --set full --export  report_name.ncu-rep ./exec_name.out
view with:
    ncu-ui report_name.ncu-rep    (opens the ui for nsight compute)
```