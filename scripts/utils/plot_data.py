import matplotlib.pyplot as plt
import csv
from datetime import datetime
import numpy as np

DATA_FILE = "/home/matthias/Projects/Autoformer/dataset/exchange_rate/exchange_rate.csv"

content = []
with open(DATA_FILE, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        if i == 0:
            col_headers = line[0].split(',')
            continue
        content.append(line[0].split(","))

content_dict = {}
for i, col in enumerate(col_headers):
    if col == "date":
        content_dict[col] = [datetime.strptime(c[i], '%Y/%m/%d %H:%M') for c in content]
        # content_dict[col] = [c[i] for c in content]
    else:
        content_dict[col] = [float(c[i]) for c in content]


def plot_raw_data(data):

    n_plots = len(data)
    n_rows = int(np.ceil(np.sqrt(n_plots)))
    n_cols = int(np.ceil(n_plots / n_rows))

    fig, axs = plt.subplots(n_rows, n_cols)
    axs = np.reshape(axs, (-1,))

    for i, col in enumerate(data):
        if col == "date":
            continue
        axs[i].plot(data["date"][-1500:], data[col][-1500:])
        axs[i].set_title(col)
    plt.show()

plot_raw_data(content_dict)
