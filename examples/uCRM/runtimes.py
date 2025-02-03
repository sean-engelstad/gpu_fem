import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# num nodes
# L1 - 23738, L2 - , L3 - 
meshes = [23738, ]

# gpu_fem GPU runtimes data, on H100 GPU's
gpu_dict = {
    # factorization is on the host anyways
    'factorization' : [n, 4.182, ],
    'assembly' : [n, 0.0172, ],
    'LU_solve' : [n, 1.983, ]
}

# 1 process
cpu_1_dict = {
    'factorization' : [0.345],
    'assembly' : [0.705],
    'LU_solve' : [3.2891]
}

# on 4 processes
cpu_4_dict = {
    'factorization' : [0.2021],
    'assembly' : [0.2752],
    'LU_solve' : [3.0511]
}