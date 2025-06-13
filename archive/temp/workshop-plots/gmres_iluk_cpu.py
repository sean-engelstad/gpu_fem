

class GMRES:
    def __init__(self, name, fill, times):
        self.name = name
        self.fill = fill
        self.times = times

# runtimes with different methods TACS CPU
# with 8 procs each
AMD_LU = 1.6643e0
ND_LU = 1.4677e0

gmres_list = [
    GMRES("AMD-GMRES",
        fill=[5, 7, 9, 11, 15, 25],
        times=[6.52, 4.7224, 3.6049, 3.6875, 2.5143, 2.3684]
    ),
    GMRES("ND-GMRES",
        fill=[5, 7, 9, 11, 15, 25],
        times=[7.9863, 5.8893, 4.6693, 3.8734, 3.1207, 2.2599]
    ),
    GMRES("RCM-GMRES",
        fill=[5, 7, 9, 11, 15, 25],
        times=[6.957, 5.1646, 4.4819, 5.2917, 5.5599, 10.144]
    ),
]