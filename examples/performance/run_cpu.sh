echo "nprocs, nxe, ndof, nz_time(s), assembly_time(s), solve_time(s), tot_time(s)"
python _cpu_baseline.py --nxe 10
python _cpu_baseline.py --nxe 20
python _cpu_baseline.py --nxe 40
python _cpu_baseline.py --nxe 80
python _cpu_baseline.py --nxe 160
python _cpu_baseline.py --nxe 320