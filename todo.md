
## Journal paper plan

- [ ] multigrid has these issues with MITC for thin shells
- [ ] show some of the beam multigrid stuff (only subset of elements cite dissertation).. (adding this in) - C1 is important
- [ ] do include plate +/or cylinder very briefly
   * then this shows multigrid good solver but not fully robust to thin shells (with MITC4), and limitations of mixed IGA methods explain.
- [ ] BDDC-LU improves performance but requires wraparound subdomain splitting
   - [ ] add multilevel BDDC dots into the cylinder + wing runtime comparisons
- [ ] empirical / theoretical evidence of wraparound method
   * run a bunch of different multi-patch structures, with wraparound fraction
- [ ] push more on the unstructured mesh BDDC (and comparison to structured BDDC in paper)
   * Dr. K said can I just add a vertex node on subdomain boundaries that are missing a vertex node (even though not 3 or more subdomain connections there). Yes this could be good solution, need robust here.
- [ ] demonstrate GPU (GPU-CPU) speedups + linear solver comparison
- [ ] optimization demonstration


## Journal paper tasks

- [ ] add README.md for gpu_fem to show highly scalable structural analysis pictures (and make people interested in using it)

1. [ ] Finish multi-GPU development
   - [ ] GMG-ASW on multi-GPU
   - [ ] 2-level BDDC-LU on multi-GPU
      - [ ] CuDSS multi-GPU Schur complement for coarse direct solve

2. [ ] finish unstructured BDDC (do need this for paper)
   - [ ] unstructured BDDC on plate/cylinder case
      * try adding subdomain vertices to subdomian interfaces with none (check), near bndry
   - [ ] unstructured wraparound BDDC on wing problem (implement pseudocode)
      * demo on uCRM and HSCT meshes as additional cases in paper (maybe a table and with pictures)

3. [ ] high DOF wing optimization cases
   - [ ] implement smeared stiffener, buckling constraints + multiple load cases
   - [ ] do very high DOF Problems with GMG-ASW vs BDDC-LU for instance (laso show single GPU results too)

4. [ ] writing
   - [ ] add brief element affect on multigrid (beam, plates, shells)
   - [ ] add multilevel BDDC to scatter plots, table (fix any issues with it also if need to)
      * or just in a separate plot comparing 2-level, 3-level and 4-level BDDC maybe (would need dev for 4-level also)
   - [ ] evidence that BDDC wraparound is good
   - [ ] comparison of unstructured + structured BDDC

6. put my GPU code into TACS repo (prob BDDC first, MITC4 shells)
   - [ ] make interface that constructs GPU assembler and classes from CPU assembler
   - [ ] then runs the GPU code as usual
   - [ ] implement BDDC (with these two tasks to make it more practical)
      - [ ] BDDC wraparound for unstructured meshes (and gen single-patch), min # corners and other metrics maybe
      - [ ] BDDC with more general simply supported vs clamped BCs (prob just duplicate node and make some DOF one in each view)


