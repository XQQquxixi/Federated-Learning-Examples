import autograd.numpy as np
from network import random_init, sigmoid, net_predict, gd_step, sample_batch
import matplotlib
import time

ITERATION = 501
NDEVICE = 4
C = 0.25
INIT_STD = 0.1
NHID = 50
BATCH = 30
DIM = 100   # dim of gaussian
NUM = 20  # number of data per gaussian
GAP = DIM // NDEVICE * NUM * 2   # number of data per client
TOTAL = NUM * DIM * 2   # total number of data (all gaussians generated)


def partition_most_easy(train_data, train_label, lr, steps):
    """partition of far gaussians"""
    clients = []
    for i in range(NDEVICE):
        total_x = []
        total_y = []
        for j in range(DIM // NDEVICE):
            x = np.concatenate((train_data[i * GAP + j * 2 * NUM: i * GAP + j * 2 * NUM + NUM],
                                train_data[((i + 2) * GAP + (j * 2 + 1) * NUM) % TOTAL:
                                           ((i + 2) * GAP + (j * 2 + 1) * NUM) % TOTAL + NUM]), axis=0)
            y = np.concatenate((train_label[i * GAP + j * 2 * NUM: i * GAP + j * 2 * NUM + NUM],
                                train_label[((i + 2) * GAP + (j * 2 + 1) * NUM) % TOTAL:
                                            ((i + 2) * GAP + (j * 2 + 1) * NUM) % TOTAL + NUM]), axis=0)
            total_x.append(x)
            total_y.append(y)
        c = Client(lr, steps, np.asarray(total_x).reshape(-1, DIM), np.asarray(total_y).reshape(-1,))
        clients.append(c)

    return clients


def partition_easy(train_data, train_label, lr, steps):
    """partition of far gaussians"""
    clients = []
    for i in range(NDEVICE):
        total_x = []
        total_y = []
        for j in range(DIM // NDEVICE):
            x = np.concatenate((train_data[i * GAP + j * 2 * NUM: i * GAP + j * 2 * NUM + NUM],
                                train_data[((i + 1) * GAP + (j * 2 + 1) * NUM) % TOTAL:
                                           ((i + 1) * GAP + (j * 2 + 1) * NUM) % TOTAL + NUM]), axis=0)
            y = np.concatenate((train_label[i * GAP + j * 2 * NUM: i * GAP + j * 2 * NUM + NUM],
                                train_label[((i + 1) * GAP + (j * 2 + 1) * NUM) % TOTAL:
                                            ((i + 1) * GAP + (j * 2 + 1) * NUM) % TOTAL + NUM]), axis=0)
            total_x.append(x)
            total_y.append(y)
        c = Client(lr, steps, np.asarray(total_x).reshape(-1, DIM), np.asarray(total_y).reshape(-1,))
        clients.append(c)

    return clients


def partition_hard(train_data, train_label, lr, steps):
    """partition of close gaussians"""
    clients = []
    for i in range(NDEVICE):
        c = Client(lr, steps, train_data[i * GAP: (i+1) * GAP], train_label[i * GAP: (i+1) * GAP])
        clients.append(c)

    return clients


def partition_random(train_data, train_label, lr, steps):
    """random partitions"""
    clients = []
    for i in range(NDEVICE):
        batch_x, batch_y = sample_batch(train_data, train_label, GAP)
        c = Client(lr, steps, batch_x, batch_y)
        clients.append(c)

    return clients


class Objective:
    """Mean squared error."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, params):
        # loss
        return np.mean(self.y * np.logaddexp(0, -net_predict(params, self.x)) \
                       + (1 - self.y) * np.logaddexp(0, net_predict(params, self.x)))


class Client:
    def __init__(self, inner_lrate, num_steps, data, label):
        self.data = data
        self.label = label
        self.inner_lrate = inner_lrate
        self.E = num_steps
        self.num = data.shape[0]

    def __call__(self, params):
        for _ in range(self.E):
            for _ in range(self.num // BATCH):
                batch_x, batch_y = sample_batch(self.data, self.label, BATCH)
                co = Objective(batch_x, batch_y)
                params = gd_step(co, params, self.inner_lrate)
        return params

    def lr_decay(self, n):
        self.inner_lrate /= n


class FedAvg:
    def __init__(self, clr, cstep):
        self.clr = clr
        self.cstep = cstep
        self.traj = []
        self.time_traj = []
        self.update_num = []

    def visualize(self, params, title, ax, data, label=None):
        colors = ['red', 'green']
        if label is not None:
            ax.scatter(data[:, 0], data[:, 1], c=label,
                       cmap=matplotlib.colors.ListedColormap(colors))
        else:
            predict = sigmoid(net_predict(params, data))
            ax.scatter(data[:, 0], data[:, 1], c=np.where(predict >= 0.5, 1, 0),
                       cmap=matplotlib.colors.ListedColormap(colors))
        ax.set_title(title)

    def weight_decay(self, clients, d):
        for c in clients:
            c.lr_decay(d)

    def train(self, train_data, train_label, test_data, test_label, mode="random", decay=250, iteration=ITERATION, d=2.0):
        # np.random.seed(0)
        params = random_init(INIT_STD, NHID)
        if mode == "easy":
            clients = partition_easy(train_data, train_label, self.clr, self.cstep)
        elif mode == "most_easy":
            clients = partition_most_easy(train_data, train_label, self.clr, self.cstep)
        elif mode == "hard":
            clients = partition_hard(train_data, train_label, self.clr, self.cstep)
        elif mode == "random":
            clients = partition_random(train_data, train_label, self.clr, self.cstep)

        total_n = train_data.shape[0]
        start_here = time.time()
        m = int(max(C * NDEVICE, 1))

        for i in range(iteration):
            start = time.time() - start_here
            # val loss
            val_loss = Objective(test_data, test_label)(params)
            if i % 50 == 0:
                print('Iteration %d val objective: %f' % (i, val_loss))
            self.time_traj.append(start)
            self.traj.append(val_loss)
            self.update_num.append(i * m * self.cstep * GAP // BATCH)

            s = list(np.random.permutation(NDEVICE))[:m]
            # list of dicts of gradient
            p_dict = {c: params for c in range(NDEVICE)}

            if i > 0 and i % decay == 0:
                self.weight_decay(clients, d)

            for c in s:
                p_dict[c] = clients[c](params)

            params = {p: 0 for p in params}
            for p in params:
                for c in range(NDEVICE):
                    weight = clients[c].num / total_n
                    params[p] += weight * p_dict[c][p]
