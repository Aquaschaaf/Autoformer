import matplotlib.pyplot as plt
import vectorbt as vbt

import numpy as np



def generate_signal(predictions, lookahead=15, exit_thresh=-0.03):
    entries = np.zeros(len(predictions), dtype=bool)
    exits = np.zeros(len(predictions), dtype=bool)

    for i in range(len(predictions)):
        if np.all(predictions[i, :] >= 0.0):  # Check if all following 10 entries are greater than 0
            entries[i] = True  # Set result to 1 if condition is met

        elif np.any(predictions[i, :] < exit_thresh):  # Check condition
        # elif np.any(predictions[i, :] < exit_thresh):  # Check condition
                exits[i] = True


    return entries, exits

def _plot_signals(history, entries, exits, sell_thresh):

    entry_data = np.empty(entries.shape).astype(np.float32)
    entry_data[entries] = history[entries]

    exit_data = np.empty(exits.shape).astype(np.float32)
    exit_data[exits] = history[exits]

    plt.figure()
    plt.plot(history.values, color="b")
    plt.scatter(list(range(len(entries))), entry_data, color="g")
    plt.scatter(list(range(len(exits))), exit_data, color="r")
    plt.title("Signals for Buy: (preds > 0.0).all() and Sell: (preds < {}).any()".format(sell_thresh))
    plt.show()

def backtest(history, pred_classes, pred_data, init_cash=1000, fees=0.0025, slippage=0.0025, use_step=10):

    entries = pred_classes[:, use_step] == 2
    exits = pred_classes[:, use_step] == 0

    max_return = -1000
    best_pf = None
    for i in range(10):
        thresh = -i/100.
        entries, exits = generate_signal(pred_data, exit_thresh=thresh)
        pf = vbt.Portfolio.from_signals(history, entries=entries, exits=exits, init_cash=init_cash, fees=fees, slippage=slippage)

        if i == 5:
            _plot_signals(history, entries, exits, thresh)

        tr = pf.total_return()
        if tr > max_return:
            max_return = tr
            best_pf = pf

    print("Best SellThresh: {}".format(thresh))
    # Print Portfolio Stats and Return Stats
    print(pf.stats())
    print(pf.returns_stats())
    best_pf.plot().show()
    plt.show()

