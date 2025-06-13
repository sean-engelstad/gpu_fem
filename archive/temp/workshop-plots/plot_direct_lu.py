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
VALID_TIME_KEYS = ['nz', 'resid', 'jac', 'fact', 'triang', 'mult', 'tot']

def plot_times(df, time_keys, all_flag):
    cpu_df = df[df['hardware'] == 'NAS_CPU']
    gpu_df = df[df['hardware'] == 'loc_GPU'].iloc[0]  # Single GPU row

    if all_flag:
        time_keys = VALID_TIME_KEYS

    for short_key in time_keys:
        key = f'{short_key}_time'

        plt.style.use(niceplots.get_style())

        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'DejaVu Sans'

        plt.figure()
        # Plot CPU line
        x = cpu_df['count'].to_numpy(); y = cpu_df[key].to_numpy()
        # print(f"{x=} {y=}")
        print(f"{key=}")
        plt.plot(x, y, marker='o', label='CPUs')
        # Plot GPU horizontal line
        plt.plot(x, [gpu_df[key] for _ in x], label='1 GPU (3060 Ti)')

        # plt.title(f'{key} vs # of CPU Procs')
        plt.xlabel('# of CPU Procs')
        plt.ylabel(f'{key} (sec)')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.margins(x=0.05, y=0.05)

        # Set custom ticks on log scale (adjust as needed)
        xticks = [2, 4, 8, 16, 32, 64]
        plt.xticks(xticks, labels=[str(t) for t in xticks])

        # custom ylim

        # # Set major ticks at 1eX and minor ticks at 2eX and 5eX
        ax = plt.gca()
        ax.set_yscale('log')

        # Major ticks at full decades (including the expanded lower decade)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=10))
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

        # # Minor ticks at 2× and 5× within each decade
        # ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=[2.0, 5.0], numticks=10))
        # ax.yaxis.set_minor_formatter(ticker.NullFormatter())


        # final plot
        plt.savefig(f'out/{key}_vs_cpu.png', dpi=400)
        # plt.show()


def plot_bar_chart(df):
    cpu_df = df[df['hardware'] == 'NAS_CPU']
    mydict = {}
    for key in VALID_TIME_KEYS:
        key2 = f"{key}_time"
        if key == "resid":
            key = "add_resid"
        elif key == "jac":
            key = "add_jac"
        elif key == "fact":
            key = "LU_fact"
        elif key == "triang":
            key = "triang_solve"
        elif key == "nz":
            key = "nz_pattern"
        mydict[key] = cpu_df[key2].to_numpy()[-2]

    print(f"{mydict=}")

    # Convert dictionary to DataFrame for seaborn
    data = pd.DataFrame({
        'Operation': list(mydict.keys()),
        'Time': list(mydict.values())
    })

    plt.figure(figsize=(8,6))
    sns.barplot(x='Operation', y='Time', data=data, palette='tab10')
    plt.xticks(rotation=45, ha='right')
    plt.title('TACS CPU Direct LU Solve')
    plt.tight_layout()
    plt.yscale('log')
    plt.ylabel("Time (s)")
    plt.margins(y=0.05, x=0.05)
    plt.savefig("out/cpu_barchart.png", dpi=400)
    # plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot CPU vs GPU times.')
    parser.add_argument('--time', choices=VALID_TIME_KEYS,
                        help='Which time field to plot (e.g., nz, resid, jac, fact, triang)')
    parser.add_argument('--all', action='store_true', help='Plot all time fields')

    args = parser.parse_args()

    df = pd.read_csv(CSV_FILE)

    if not args.all and not args.time:
        parser.error("You must specify either --time or --all.")

    time_keys = [args.time] if args.time else []
    plot_times(df, time_keys, args.all)

    # now plot pie chart
    plot_bar_chart(df)

if __name__ == '__main__':
    main()
