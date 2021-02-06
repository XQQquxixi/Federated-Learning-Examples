import autograd.numpy as np
import autograd as ag
from network import random_init, sigmoid, net_predict, sample_batch
import matplotlib
import time

ITERATION = 601
NDEVICE = 4
C = 0.25
LRATE = 0.5
INIT_STD = 0.1
NHID = 50
DIM = 100   # dim of gaussian
NUM = 20  # number of data per gaussian
GAP = DIM // NDEVICE * NUM * 2   # number of data per client
TOTAL = NUM * DIM * 2   # total number of data (all gaussians generated)


def gd_step(cost, params):
    """return gradients rather than parameters"""
    grad_cost = ag.grad(cost)
    gradient = grad_cost(params)
    return gradient


def partition_easy(train_data, train_label):
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
        c = Client(np.asarray(total_x).reshape(-1, DIM), np.asarray(total_y).reshape(-1,))
        clients.append(c)

    return clients


def partition_hard(train_data, train_label):
    """partition of close gaussians"""
    clients = []
    for i in range(NDEVICE):
        c = Client(train_data[i * GAP: (i+1) * GAP], train_label[i * GAP: (i+1) * GAP])
        clients.append(c)
    return clients


def partition_random(train_data, train_label):
    """random partition"""
    clients = []
    for i in range(NDEVICE):
        batch_x, batch_y = sample_batch(train_data, train_label, GAP)
        c = Client(batch_x, batch_y)
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
               + (1-self.y) * np.logaddexp(0, net_predict(params, self.x)))


class Client:
    """ Full batch gradient descent on the inner objective."""
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.num = data.shape[0]

    def __call__(self, params):
        """Compute the meta-objective"""
        co = Objective(self.data, self.label)
        gradient = gd_step(co, params)

        return gradient


class FedSGD:
    """Full batch (non-stochastic) gradient descent: FedSGD """
    def __init__(self):
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

    def train(self, train_data, train_label, test_data, test_label, mode="random", iter=ITERATION):
        params = random_init(INIT_STD, NHID)

        if mode == "easy":
            clients = partition_easy(train_data, train_label)
        elif mode == "hard":
            clients = partition_hard(train_data, train_label)
        elif mode == "random":
            clients = partition_random(train_data, train_label)

        total_n = train_data.shape[0]
        lr = LRATE
        start_here = time.time()

        for i in range(iter):
            start = time.time() - start_here
            # val loss
            val_loss = Objective(test_data, test_label)(params)
            if i % 50 == 0:
                print('Iteration %d val objective: %f' % (i, val_loss))
            self.time_traj.append(start)
            self.traj.append(val_loss)

            m = int(max(C * NDEVICE, 1))
            self.update_num.append(i * m)
            s = list(np.random.permutation(NDEVICE))[:m]
            # list of dicts of gradient
            gradient = {p: 0 for p in params}
            g_dict = {c: gradient for c in range(NDEVICE)}
            for c in s:
                g_dict[c] = clients[c](params)

            # aggregate gradient
            for p in gradient:
                for c in s:
                    weight = clients[c].num / total_n
                    gradient[p] += weight * g_dict[c][p]
            # weight decay
            if i > 0 and i % 200 == 0:
                lr /= 2.

            for p in params:
                params[p] -= lr * gradient[p]
