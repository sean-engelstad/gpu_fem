from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT

# Change `use_sfmath` to False to use computer modern
x = XDSM(use_sfmath=True)

# first version of the xdsm script (unoptimized for memory storage plan)
# will produce another one with optimized memory storage plan

# define subsystems
x.add_system("main", OPT, r"\text{ShellResidual}")
x.add_system("NN1", FUNC, "NodeNorm1")
x.add_system("NN2", FUNC, "NodeNorm2")
x.add_system("TR", FUNC, "ShellTransform")
x.add_system("DS", FUNC, "DrillStrain")
x.add_system("CD", FUNC, "Director")
x.add_system("TS", FUNC, "TyingStrain")
x.add_system("PT", FUNC, "GetQuadPt")
x.add_system("ITS", FUNC, "InterpTyingStrain")
x.add_system("NBAS", FUNC, "InterpNormals")
x.add_system("XBAS", FUNC, "InterpXpts")
x.add_system("UBAS", FUNC, "InterpUVars")
x.add_system("DBAS", FUNC, "InterpDirector")
x.add_system("DSBAS", FUNC, "InterpDrillBasis")
x.add_system("DG", FUNC, "DispGrad")
x.add_system("RF", FUNC, "RotateFrame")
x.add_system("WR", SOLVER, "weakRes")
x.add_system("STRAIN", FUNC, "ShellStrain")
x.add_system("STRESS", FUNC, "ShellStress")
x.add_system("ENERGY", FUNC, "StrainEnergy")
x.add_system("WR_Sens", FUNC, "weakResSens")
x.add_system("DG_Sens", FUNC, "DispGradSens")
x.add_system("RF_Sens", FUNC, "RotateFrameSens")
x.add_system("ITS_TR", FUNC, "InterpTyingStrainTR")
x.add_system("DSBAS_TR", FUNC, "InterpDrillTR")
x.add_system("UBAS_TR", FUNC, "InterpUVarsTR")
x.add_system("DBAS_TR", FUNC, "InterpDirectorTR")
x.add_system("DS_Sens", FUNC, "DrillStrainSens")
x.add_system("TS_Sens", FUNC, "TyingStrainSens")
x.add_system("CD_Sens", FUNC, "DirectorSens")
x.add_system("F_RES", SOLVER, "FinalResidual")


# node normals
x.connect("main", "NN1", "xpts[12]")
x.connect("main", "NN2", "xpts[12]")

# compute transform
x.connect("main", "TR", "physData[7]")

# drill strain
x.connect("main", "DS", "vars[24]")
x.connect("TR", "DS", "Tmat[9]")
x.connect("NN2", "DS", "Xdn[36]")

# compute director
x.connect("main", "CD", "vars[24]")
x.connect("NN1", "CD", "fn[12]")

# compute tying strain
x.connect("main", "TS", "xpts[12], vars[24]")
x.connect("NN1", "TS", "fn[12]")
x.connect("CD", "TS", "d[12]") 

# get quadrature point
x.connect("main", "PT", "iquad[int]")

# interp tying strain
x.connect("PT", "ITS", "pt[2]")
x.connect("TS", "ITS", "ety[9]")

# interp normals
x.connect("PT", "NBAS", "pt[2]")
x.connect("NN1", "NBAS", "fn[12]")

# interp xpt basis
x.connect("PT", "XBAS", "pt[2]")
x.connect("main", "XBAS", "xpts[12]")

# interp U vars basis
x.connect("PT", "UBAS", "pt[2]")
x.connect("main", "UBAS", "vars[24]")

# interp D vars basis (director)
x.connect("PT", "DBAS", "pt[2]")
x.connect("CD", "DBAS", "d[12]")

# shell compute disp grad
x.connect("NBAS", "DG", "n0[3],nxi[3],neta[3]")
x.connect("XBAS", "DG", "Xxi[3],Xeta[3]")
x.connect("UBAS", "DG", "u0xi[3],u0eta[3]")
x.connect("DBAS", "DG", "d0[3],d0xi[3],d0eta[3]")
x.connect("TR", "DG", "Tmat[9]")

# sym rotate frame
x.connect("DG", "RF", "XdinvT[9]")
x.connect("ITS", "RF", "gty[6]")

# interp drill basis
x.connect("DS", "DSBAS", "etn[4]")
x.connect("PT", "DSBAS", "pt[2]")

# physics weak residual
# just a solver block, no direct connections yet
x.connect("STRAIN", "WR", "ForwStack")
x.connect("STRESS", "WR", "ForwStack")
x.connect("ENERGY", "WR", "ForwStack")

# shell strain
x.connect("DG", "STRAIN", "u0x[9],u1x[9]")
x.connect("RF", "STRAIN", "e0ty[6]")
x.connect("DSBAS", "STRAIN", "et[1]")

# shell stress
x.connect("main", "STRESS", "physData[7]")
x.connect("STRAIN", "STRESS", "E[9]")

# strain energy
x.connect("STRAIN", "ENERGY", "E[9]")
x.connect("STRESS", "ENERGY", "S[9]")

# weak res sens
x.connect("WR_Sens", "WR", "RevStack")

# disp grad sens
x.connect("WR_Sens", "DG_Sens", r"\overline{u0x}[9],\overline{u1x}[9]")
x.connect("NBAS", "DG_Sens", "n0[3],nxi[3],neta[3]")
x.connect("XBAS", "DG_Sens", "Xxi[3],Xeta[3]")
x.connect("UBAS", "DG_Sens", "u0xi[3],u0eta[3]")
x.connect("DBAS", "DG_Sens", "d0[3],d0xi[3],d0eta[3]")
x.connect("TR", "DG_Sens", "Tmat[9]")

# transpose interpolation sensitivities
x.connect("WR_Sens", "DSBAS_TR", r"\overline{et}[1]")

# Sym mat rotate frame
x.connect("DG_Sens", "RF_Sens", "XdinvT[9]")
x.connect("WR_Sens", "RF_Sens", r"\overline{e0ty}[6]")

# interp tying strain transpose
x.connect("RF_Sens", "ITS_TR", r"\overline{gty}[6]")
x.connect("PT", "ITS_TR", "pt[2]")

# interp drill TR
x.connect("WR_Sens", "DSBAS_TR", r"\overline{et}[1]")

# interp uvars TR
x.connect("DG_Sens", "UBAS_TR", r"\overline{u0xi}[3],\overline{u0eta}[3]")

# interp director TR
x.connect("DG_Sens", "DBAS_TR", r"\overline{d0}[3]")

# drill strain sens
x.connect("NN1", "DS_Sens", "fn[12]")
x.connect("NN2", "DS_Sens", "Xdn[36]")
x.connect("DSBAS_TR", "DS_Sens", r"\overline{etn}[4]")

# compute tying strain sens
x.connect("main", "TS_Sens", "xpts[12]")
x.connect("NN1", "TS_Sens", "fn[12]")
x.connect("TR", "TS_Sens", "Tmat[9]")
x.connect("ITS_TR", "TS_Sens", r"\overline{ety}[9]")

# compute director sens
x.connect("TS_Sens", "CD_Sens", r"\overline{d}[12]")
x.connect("DBAS_TR", "CD_Sens", r"\overline{d}[12]")

# final residual
x.connect("main", "F_RES", "res[24]")
x.connect("UBAS_TR", "F_RES", "res[24]")
x.connect("TS_Sens", "F_RES", "res[24]")
x.connect("CD_Sens", "F_RES", "res[24]")


# x.add_system("DSBAS_TR", FUNC, "InterpDrillTR")
# x.add_system("UBAS_TR", FUNC, "InterpUVarsTR")
# x.add_system("DBAS_TR", FUNC, "InterpDVarsTR")
# x.add_system("DS_Sens", FUNC, "DrillStrainSens")
# x.add_system("TS_Sens", FUNC, "TyingStrainSens")
# x.add_system("CD_Sens", FUNC, "DirectorSens")
# x.add_system("F_RES", SOLVER, "FinalResidual")

# final write
x.write("1_xdsm_v1")