import matplotlib.pyplot as plt
import csv
from datetime import datetime
import numpy as np

from evaluate.utils.helpers import get_num_row_cols

def find_date_format(example_date):
    date_fmt_candidates = ['%Y/%m/%d %H:%M', '%Y-%m-%d']
    date_format = None
    for date_fmt_candidate in date_fmt_candidates:
        try:
            datetime.strptime(example_date, date_fmt_candidate)
            print("Matched date format {}".format(date_fmt_candidate))
            date_format = date_fmt_candidate
        except:
            pass
    return date_format


def load_data(data_file):

    content = []
    with open(data_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            if i == 0:
                col_headers = line[0].split(',')
                continue
            content.append(line[0].split(","))

    content_dict = {}
    for i, col in enumerate(col_headers):
        if col == "date":
            # date_format = find_date_format()
            date_format = find_date_format(content[0][i])
            assert date_format is not None, "Date format could not be matched"
            content_dict[col] = [datetime.strptime(c[i], date_format) for c in content]
            # content_dict[col] = [c[i] for c in content]
        else:
            content_dict[col] = [float(c[i]) for c in content]

    return content_dict

def plot_raw_data(data):

    dates = data["date"]
    del data["date"]

    n_rows, n_cols = get_num_row_cols(len(data))

    fig, axs = plt.subplots(n_rows, n_cols)
    axs = np.reshape(axs, (-1,))

    for i, col in enumerate(data):
        if col == "date":
            continue
        axs[i].plot(dates[-1500:], data[col][-1500:])
        axs[i].set_title(col)
    plt.show()


if __name__ == "__main__":
    DATA_FILE = "/home/matthias/Projects/Autoformer/dataset/btc_usd/btc_usd.csv"
    content_dict = load_data(DATA_FILE)
    plot_raw_data(content_dict)

