import numpy as np
import matplotlib.pyplot as plt

input = np.load("tmp/input.npy")
all_outputs = np.load("tmp/all_outputs.npy")
mean = np.load("tmp/mean.npy")
std = np.load("tmp/std.npy")
true = np.load("tmp/true.npy")

all_outputs = all_outputs.reshape((all_outputs.shape[0], -1, all_outputs.shape[2], all_outputs.shape[3]))
input = input.reshape(-1, input.shape[-2], input.shape[-1])
mean = mean.reshape(-1, mean.shape[-2], mean.shape[-1])
std = std.reshape((-1, std.shape[-2], std.shape[-1]))
true = true.reshape((-1, true.shape[-2], true.shape[-1]))


num_batches = input.shape[0]
seq_length = input.shape[1]
pred_length = mean.shape[1]

fig = plt.figure()
x = list(range(0, seq_length+pred_length))
for batch_idx in range(num_batches):
    for i in range(all_outputs.shape[0]):
        plt.plot(x[seq_length:], all_outputs[i, batch_idx, :, -1], alpha=0.25)
    plt.plot(x[:seq_length], input[batch_idx, :, -1])
    plt.plot(x[seq_length:], mean[batch_idx, :, -1], label="Mean Prediction")
    plt.plot(x[seq_length:], true[batch_idx, :, -1], label="Ground Truth")

    plt.fill_between(x[seq_length:], mean[batch_idx, :, -1] - std[batch_idx, :, -1], mean[batch_idx, :, -1] + std[batch_idx, :, -1])

    plt.legend()
    plt.show()