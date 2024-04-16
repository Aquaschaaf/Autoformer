import vectorbt as vbt
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def percentage_change(data):
    # Calculate percentage change from the first entry
    return (data - data[0]) / data[0]


def generate_gt_signal(close, lookahead=15, exit_thresh=-0.05):
    entries = np.zeros_like(close, dtype=bool)  # Initialize result array with zeros
    for i in range(len(close)):
        if i + lookahead < len(close):  # Ensure there are at least 10 following entries
            pct = percentage_change(close[i:i + lookahead])
            if np.all(pct >= 0.0):  # Check if all following 10 entries are greater than 0
                entries[i] = True  # Set result to 1 if condition is met

    exits = np.zeros_like(close, dtype=bool)  # Initialize result array with False
    for i in range(len(close)):
        if i + lookahead < len(close):  # Ensure there are at least 10 following entries
            pct = percentage_change(close[i:i + lookahead])
            if np.any(pct < exit_thresh) or pct[-1] < -0.01:  # Check condition
                exits[i] = True

    entry_data = [c if entries[i] else None for i, c in enumerate(close)]
    exit_data = [c if exits[i] else None for i, c in enumerate(close)]

    plt.figure()
    plt.plot(close, color="b")
    plt.scatter(list(range(len(entry_data))), entry_data, color="g")
    plt.scatter(list(range(len(exit_data))), exit_data, color="r")
    plt.show()

    return entries, exits

data = pd.read_csv("/home/matthias/Projects/Autoformer/dataset/btc_usd/BTC-Hourly.csv")
close = data["Close"]

entries, exits = generate_gt_signal(close.values)
pf = vbt.Portfolio.from_signals(close, entries=entries, exits=exits)

pf.plot().show()