import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

plt.rcParams["image.interpolation"] = "nearest"  # don't interpolate: show square pixels
plt.rcParams[
    "image.cmap"
] = "gray"  # use grayscale output rather than a (potentially misleading) color heatmap


def plot_loss_and_acc(loss_and_acc_dict):
    plt.figure()
    tmp = list(loss_and_acc_dict.values())
    maxEpoch = len(tmp[0][0])

    maxLoss = max([max(x[0]) for x in loss_and_acc_dict.values()]) + 0.1
    minLoss = max(0, min([min(x[0]) for x in loss_and_acc_dict.values()]) - 0.1)

    for name, lossAndAcc in loss_and_acc_dict.items():
        plt.plot(range(1, 1 + maxEpoch), lossAndAcc[0], "-s", label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(range(0, maxEpoch + 1, 2))
    plt.axis([0, maxEpoch, minLoss, maxLoss])
    plt.show()

    maxAcc = min(1, max([max(x[1]) for x in loss_and_acc_dict.values()]) + 0.1)
    minAcc = max(0, min([min(x[1]) for x in loss_and_acc_dict.values()]) - 0.1)

    plt.figure()

    for name, lossAndAcc in loss_and_acc_dict.items():
        plt.plot(range(1, 1 + maxEpoch), lossAndAcc[1], "-s", label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(range(0, maxEpoch + 1, 2))
    plt.axis([0, maxEpoch, minAcc, maxAcc])
    plt.legend()
    plt.show()


def plot_loss_and_acc_save(loss_and_acc_dict, file_name):
    plt.figure()
    tmp = list(loss_and_acc_dict.values())
    maxEpoch = len(tmp[0][0])

    maxLoss = max([max(x[0]) for x in loss_and_acc_dict.values()]) + 0.1
    minLoss = max(0, min([min(x[0]) for x in loss_and_acc_dict.values()]) - 0.1)

    for name, lossAndAcc in loss_and_acc_dict.items():
        plt.plot(range(1, 1 + maxEpoch), lossAndAcc[0], "-s", label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(range(0, maxEpoch + 1, 2))
    plt.axis([0, maxEpoch, minLoss, maxLoss])
    plt.savefig(file_name + "-Loss.png")
    plt.close()

    maxAcc = min(1, max([max(x[1]) for x in loss_and_acc_dict.values()]) + 0.1)
    minAcc = max(0, min([min(x[1]) for x in loss_and_acc_dict.values()]) - 0.1)

    plt.figure()

    for name, lossAndAcc in loss_and_acc_dict.items():
        plt.plot(range(1, 1 + maxEpoch), lossAndAcc[1], "-s", label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(range(0, maxEpoch + 1, 2))
    plt.axis([0, maxEpoch, minAcc, maxAcc])
    plt.legend()
    plt.savefig(file_name + "-Accuracy.png")
    plt.close()


def vis_square(data, filename):
    """Take an array of shape (m , n, height, width) and visualize each (height, width) thing in a grid of size m * n
    """
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    m = data.shape[0]
    n = data.shape[1]
    padding = ((0, 0), (0, 0), (1, 1), (1, 1))  # add some space between filters
    data = np.pad(
        data, padding, mode="constant", constant_values=1
    )  # pad with ones (white)
    # tile the filters into an image
    data = data.transpose((0, 2, 1, 3))
    data = data.reshape((m * data.shape[1], n * data.shape[3]))
    plt.axis("off")
    plt.imshow(data)
    plt.savefig(filename + "-Visualize.png")
    plt.close()


if __name__ == "__main__":
    filter = np.random.randint(0, 256, (3, 6, 3, 3))
    loss = [x for x in range(10, 0, -1)]
    acc = [x / 10.0 for x in range(0, 10)]
    plot_loss_and_acc_save({"as": [loss, acc]}, "lala")
    vis_square(filter, "filter")
