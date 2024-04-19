import vectorbt as vbt
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



data = pd.read_csv("/home/matthias/Projects/Autoformer/dataset/btc_usd/BTC-Hourly.csv")
close = data["Close"]

entries = np.ones_like(close).astype(bool)
exits = np.zeros_like(close).astype(bool)

best_tr = -1000000
best_tp_stop = 0
best_sl_stop = 0
best_pf = None
for i in range(1, 30):
    for j in range(1, 30):
        sl_thresh = i/100.
        tp_thresh = j/100.
        pf = vbt.Portfolio.from_signals(close, entries=entries, exits=exits, sl_stop=sl_thresh, tp_stop=tp_thresh)

        tr = pf.total_return()

        if tr > best_tr:
            best_tr = tr
            best_pf = pf
            best_tp_stop = tp_thresh
            best_sl_stop = sl_thresh

print("Best tr: {}, Best_sl: {}, Best tp: {}".format(best_tr, best_sl_stop, best_tp_stop))
best_pf.plot().show()