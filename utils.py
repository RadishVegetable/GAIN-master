import numpy as np
import torch


def load_dataset(dataset_dir, batch_size, train_rate, miss_rate, hint_rate):
    data = np.loadtxt(dataset_dir, delimiter=",", skiprows=1)  # The data shape is [4601, 57]
    length = len(data)
    dim = len(data[0, :])  # Number of features
    scaler = StandardScaler(dim)  # provide (inverse) normalization operation
    data = scaler.transform(data)  # normalization, MIN_MAX / Z-SCORE
    # train and test division
    idx = np.random.permutation(length)
    train_length = int(length * train_rate)
    trainX = data[idx[: train_length], :]
    testX = data[idx[train_length:], :]
    train_data = Dataloader(trainX, batch_size, miss_rate, hint_rate)  # provide x,m,h,z
    test_data = Dataloader(testX, batch_size, miss_rate, hint_rate)
    return train_data, test_data, scaler, dim


class Dataloader(object):
    def __init__(self, xs, batch_size, miss_rate, hint_rate):
        self.batch_size = batch_size
        self.length = len(xs)
        self.xs = torch.tensor(xs, dtype=torch.float32)
        self.m = self.generate_random_matrix(xs.shape, miss_rate)
        b = self.generate_random_matrix(xs.shape, hint_rate)
        self.h = b * self.m + 0.5 * (1. - b)

    # generate random matrix according to the rate
    def generate_random_matrix(self, shape, rate):
        M = torch.rand(shape, dtype=torch.float32)
        M = M > rate
        M = 1. * M
        return M

    def get(self):
        idx = np.random.permutation(self.length)[: self.batch_size]  # provide a random set of data in each epoch
        x = self.xs[idx, :]
        m = self.m[idx, :]
        h = self.h[idx, :]
        z = torch.rand(x.shape) * 0.01  # 0 - 0.01 random number
        return x, m, h, z


class StandardScaler:
    def __init__(self, dim):
        self.dim = dim
        # self.mean = np.zeros(dim)
        # self.std = np.zeros(dim)
        self.Min_val = np.zeros(dim)
        self.Max_val = np.zeros(dim)

    def transform(self, data):
        for i in range(self.dim):
            # self.mean[i] = np.mean(data[:, i])
            # self.std[i] = np.std(data[:, i])
            # data[:, i] = (data[:, i] - self.mean[i]) / self.std[i] + 1e-6
            self.Min_val[i] = np.min(data[:, i])
            data[:, i] = data[:, i] - np.min(data[:, i])
            self.Max_val[i] = np.max(data[:, i])
            data[:, i] = data[:, i] / (np.max(data[:, i] + 1e-6))
        return data

    # inverse normalization
    def inverse_transform(self, data):
        for i in range(self.dim):
            # data[:, i] = (data[:, i] - 1e-6) * self.std[i] + self.mean[i]
            data[:, i] = data[:, i] * (self.Max_val[i] + 1e-6) + self.Min_val[i]
        return data
