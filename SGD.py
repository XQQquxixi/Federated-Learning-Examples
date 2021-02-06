import autograd.numpy as np
import matplotlib
from network import random_init, sigmoid, net_predict, gd_step, sample_batch
import time

INIT_STD = 0.1
NHID = 50
ITERATION = 1501
LRATE = 0.1
BATCH = 30


def visualize(params, title, ax, data, label=None):
    colors = ['red', 'green']
    if label is not None:
        ax.scatter(data[:, 0], data[:, 1], c=label,
                   cmap=matplotlib.colors.ListedColormap(colors))
    else:
        predict = sigmoid(net_predict(params, data))
        ax.scatter(data[:, 0], data[:, 1], c=np.where(predict >= 0.5, 1, 0),
                   cmap=matplotlib.colors.ListedColormap(colors))
    ax.set_title(title)


class Objective:
    """Mean squared error."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, params):
        # loss
        return np.mean(self.y * np.logaddexp(0, -net_predict(params, self.x)) \
               + (1-self.y) * np.logaddexp(0, net_predict(params, self.x)))


class SGD:
    def __init__(self):
        self.traj = []
        self.time_traj = []
        self.update_num = []

    def train(self, train_data, train_label, test_data, test_label):
        params = random_init(INIT_STD, NHID)
        lr = LRATE
        start_here = time.time()

        for i in range(ITERATION):
            start = time.time() - start_here
            # val loss
            val_loss = Objective(test_data, test_label)(params)
            if i % 50 == 0:
                print('Iteration %d val objective: %f' % (i, val_loss))
            self.traj.append(val_loss)
            self.time_traj.append(start)
            self.update_num.append(i)

            batch_data, batch_label = sample_batch(train_data, train_label, BATCH)
            co = Objective(batch_data, batch_label)

            # weight decay
            if i > 0 and i % 350 == 0:
                lr /= 2.

            params = gd_step(co, params, lr)
