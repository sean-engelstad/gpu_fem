from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT

# Change `use_sfmath` to False to use computer modern
x = XDSM(use_sfmath=True)

"""
From v1 to v2 of the XDSM,
only the variables that are stored in the outer run script will be shown now.

The variables and blocks that were removed from the outer run script storage are:

- NN2 block and Xdn[36] : large data, only was used in the drill strain sens in each case, just recompute in both
    drill strain blocks.. now have to take in Xpts and basis into each of these routines
- Shell Transform and Tmat[9] or Tmatn[36] : lot of temp data to store, just recompute this in each of the four methods
    now have to take in Data[7] and compute Tmat in each of these routines
- Basis interpolations at quad pt level for DispGrad computation;
    now disp grad routine takes in Basis and xpts[12], vars[24], fn[12], d[12] inputs directly
    we also eliminate all the extra interp routines in the main function
    - includes n0[3], n0xi[3], n0eta[3] for fn[12] interpolations
    - includes u0xi[3], u0eta[3] for vars[24] interp
    - includes d0[3], d0xi[3], d0eta[3] from d[12] interp
    - includes Xxi[3], Xeta[3] from xpts[12] interp
- Also removed interp tying strains & transpose from outer calls, so ety[9] needs to be input into 
    each disp grad and disp grad sens routine
- Include the sym mat rotate frame in DispGrad and DispGradSens; 
  this way the XdinvT doesn't need to be exposed in the main routine.
    - disp grad routines now have to convert gty[6] to e0ty[6] also.
    - want to reorganize disp grad routine to have several subcalls so a bit cleaner

"""


# first version of the xdsm script (unoptimized for memory storage plan)
# will produce another one with optimized memory storage plan

# define subsystems
x.add_system("main", OPT, r"\text{ShellResidual}")
x.add_system("NN1", FUNC, "NodeNorm1")
x.add_system("PT", FUNC, "GetQuadPt")
x.add_system("DS", FUNC, "DrillStrain")
x.add_system("CD", FUNC, "Director")
x.add_system("TS", FUNC, "TyingStrain")
x.add_system("DG", FUNC, "DispGrad")
x.add_system("WR", SOLVER, "weakRes")
x.add_system("STRAIN", FUNC, "ShellStrain")
x.add_system("STRESS", FUNC, "ShellStress")
x.add_system("ENERGY", FUNC, "StrainEnergy")
x.add_system("WR_Sens", FUNC, "weakResSens")
x.add_system("DG_Sens", FUNC, "DispGradSens")
x.add_system("DS_Sens", FUNC, "DrillStrainSens")
x.add_system("TS_Sens", FUNC, "TyingStrainSens")
x.add_system("CD_Sens", FUNC, "DirectorSens")
x.add_system("F_RES", SOLVER, "FinalResidual")


# node normals
x.connect("main", "NN1", "xpts[12]")

# get quadrature point
x.connect("main", "PT", "iquad[int]")

# drill strain
x.connect("main", "DS", "Data[7], xpts[12], vars[24]")
x.connect("NN1", "DS", "fn[12]")
x.connect("PT", "DS", "pt[2]")

# compute director
x.connect("main", "CD", "vars[24]")
x.connect("NN1", "CD", "fn[12]")

# compute tying strain
x.connect("main", "TS", "xpts[12], vars[24]")
x.connect("NN1", "TS", "fn[12]")
x.connect("CD", "TS", "d[12]") 

# shell compute disp grad
x.connect("PT", "DG", "pt[2]")
x.connect("TS", "DG", "ety[9]")
x.connect("main", "DG", "Data[7],xpts[12],vars[24]")
x.connect("NN1", "DG", "fn[12]")
x.connect("CD", "DG", "d[12]")

# physics weak residual
# just a solver block, no direct connections yet
x.connect("STRAIN", "WR", "ForwStack")
x.connect("STRESS", "WR", "ForwStack")
x.connect("ENERGY", "WR", "ForwStack")

# shell strain
x.connect("DG", "STRAIN", "u0x[9],u1x[9],e0ty[6]")
x.connect("DS", "STRAIN", "et[1]")

# shell stress
x.connect("main", "STRESS", "Data[7]")
x.connect("STRAIN", "STRESS", "E[9]")

# strain energy
x.connect("STRAIN", "ENERGY", "E[9]")
x.connect("STRESS", "ENERGY", "S[9]")

# weak res sens
x.connect("WR_Sens", "WR", "RevStack")

# disp grad sens
# TODO : should take out d[12] inputs, and some other inputs in this method
x.connect("main", "DG_Sens", "Data[7],xpts[12]")
x.connect("NN1", "DG_Sens", "fn[12]")
x.connect("WR_Sens", "DG_Sens", r"\overline{u0x}[9],\overline{u1x}[9],\overline{e0ty}[6]")
x.connect("PT", "DG_Sens", "pt[2]")

# drill strain sens
x.connect("main", "DS_Sens", "xpts[12]")
x.connect("NN1", "DS_Sens", "fn[12]")
x.connect("WR_Sens", "DS_Sens", r"\overline{et}[1]")

# compute tying strain sens
x.connect("main", "TS_Sens", "xpts[12]")
x.connect("NN1", "TS_Sens", "fn[12]")
x.connect("DG_Sens", "TS_Sens", r"\overline{ety}[9]")

# compute director sens
x.connect("TS_Sens", "CD_Sens", r"\overline{d}[12]")
x.connect("DG_Sens", "CD_Sens", r"\overline{d}[12]")

# final residual
x.connect("main", "F_RES", "res[24]")
x.connect("DS_Sens", "F_RES", "res[24]")
x.connect("DG_Sens", "F_RES", "res[24]")
x.connect("TS_Sens", "F_RES", "res[24]")
x.connect("CD_Sens", "F_RES", "res[24]")


# final outputs
x.add_output("main", "Data[7],xpts[12],vars[24]", side=LEFT)
x.add_output("NN1", "fn[12]", side=LEFT)
x.add_output("DS", "et[1]", side=LEFT)
x.add_output("CD", "d[12]", side=LEFT)
x.add_output("TS", "ety[9]", side=LEFT)
x.add_output("PT", "pt[2]", side=LEFT)
x.add_output("DG", "u0x[9], u1x[9], e0ty[6]", side=LEFT)
x.add_output("STRAIN", "E[9]", side=LEFT)
x.add_output("STRESS", "S[9]", side=LEFT)
x.add_output("ENERGY", "Uelem[1]", side=LEFT)
x.add_output("WR_Sens", r"\overline{u0x}[9],\overline{u1x}[9],\overline{e0ty}[6],\overline{et}[1]", side=LEFT)
x.add_output("DG_Sens", r"\overline{ety}[9]", side=LEFT)
x.add_output("TS_Sens", r"\overline{d}[12]", side=LEFT)
x.add_output("F_RES", "res[24]", side=LEFT)

# final write
x.write("2_xdsm_v2")