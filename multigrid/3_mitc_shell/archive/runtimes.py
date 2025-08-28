import numpy as np
import matplotlib.pyplot as plt


class RuntimeGroup:
    def __init__(self, gpu:str, case:str, solver:str, SR:float, times:list):
        assert gpu in ['3060Ti', 'A100']
        assert case in ['plate', 'cylinder']
        assert solver in ['direct', 'gmg']

        self.gpu = gpu
        self.case = case
        self.times = times

nelems = np.array([32, 64, 128, 256, 512, 1024, 2048])
ndof = nelems**2

runtime_groups = []

# reported here are just solve times (not startup / assembly, though assembly + startup slower for direct solves)
runtime_groups += [
    RuntimeGroup(
        gpu='3060Ti',
        case='plate',
        solver='direct',
        SR=50.0,
        times=[5.10e-2, 1.82e-1, 7.95e-1, 3.33e0, 1.75e1] + [np.nan] * 2,
    )
]

runtime_groups += [
    RuntimeGroup(
        gpu='3060Ti',
        case='plate',
        solver='direct',
        SR=50.0,
        times=[5.10e-2, 1.82e-1, 7.95e-1, 3.33e0, 1.75e1] + [np.nan] * 2,
    )
]