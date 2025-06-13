import pandas as pd
import matplotlib.pyplot as plt
import argparse
import niceplots
import numpy as np
import seaborn as sns

import matplotlib.ticker as ticker

# Set the CSV file path directly
CSV_FILE = 'direct_lu_ucrm.csv'  # Replace with your actual path

# Valid short names for time keys
VALID_TIME_KEYS = ['nz', 'resid', 'jac', 'fact', 'triang', 'mult']

df = df = pd.read_csv(CSV_FILE)

cpu_df = df[df['hardware'] == 'NAS_CPU']
print(f"{cpu_df=}")
arr = cpu_df.to_numpy()[:,2:-1]
print(f"{arr=}")

tot_times = np.sum(arr[:,1:], axis=1)
print(f"{tot_times=}")