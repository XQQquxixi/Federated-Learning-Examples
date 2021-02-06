import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from fedavg import FedAvg
from SGD import SGD
from FedSGD import FedSGD


def gaussian_square(a, num):
    # a > 1
    mean1 = (0, 0)
    cov = [[1, 0], [0, 1]]
    x1 = np.random.multivariate_normal(mean1, cov, (num,))

    mean2 = (0, a)
    x2 = np.random.multivariate_normal(mean2, cov, (num,))

    mean3 = (a*a, 0)
    x3 = np.random.multivariate_normal(mean3, cov, (num,))

    mean4 = (a*a, a)
    x4 = np.random.multivariate_normal(mean4, cov, (num,))

    data = np.concatenate((x1, x2), axis=0)
    data = np.concatenate((data, x3), axis=0)
    data = np.concatenate((data, x4), axis=0)

    label = np.concatenate((np.ones((num,)), np.zeros((num,))), axis=0)
    label = np.concatenate((label, label), axis=0)

    return data, label


def high_dim_gaussian(dim, epsilon, distance, num):
    cov = np.eye(dim)
    data = []
    label = []
    for i in range(dim):
        this_e = np.zeros((dim,))
        delta = np.zeros((dim,))
        this_e[i] = distance
        delta[i] = epsilon
        neg_mean = this_e - delta
        pos_mean = this_e + delta
        neg_x = np.random.multivariate_normal(neg_mean, cov, (num,))
        pos_x = np.random.multivariate_normal(pos_mean, cov, (num,))
        x = np.concatenate((neg_x, pos_x), axis=0)
        y = np.concatenate((np.zeros((num,)), np.ones((num,))), axis=0)
        data.append(x)
        label.append(y)

    return np.asarray(data).reshape(-1, dim), np.asarray(label).reshape(-1,)


def pair_distance(centers):
    c1 = centers[0]
    c2 = centers[1]
    c3 = centers[2]
    c4 = centers[3]
    print("1 and 2: %f" % np.dot(c1-c2, c1-c2))
    print("1 and 3: %f" % np.dot(c1-c3, c1-c3))
    print("1 and 4: %f" % np.dot(c1-c4, c1-c4))
    print("2 and 3: %f" % np.dot(c2-c3, c2-c3))
    print("2 and 4: %f" % np.dot(c2-c4, c2-c4))
    print("3 and 4: %f" % np.dot(c3-c4, c3-c4))


def visualize(train_data):
    pca = PCA(n_components=2)
    x = pca.fit_transform(train_data)

    co = np.repeat([['red', 'green', 'blue', 'pink', 'yellow', 'black']], 400, axis=1)
    plt.scatter(x[:, 0], x[:, 1], c=co[0])
    plt.show()


if __name__ == "__main__":
    # train_data, train_label = gaussian_square(3, 400)
    # val_data, val_label = gaussian_square(3, 100)
    train_data, train_label = high_dim_gaussian(100, 2, 10, 20)
    val_data, val_label = high_dim_gaussian(100, 2, 10, 20)

    print("fedavg start =========")
    fedavg_rd = FedAvg(0.01, 2)
    fedavg_rd.train(train_data, train_label, val_data, val_label, "random", 250, 501)

    fedavg_easy = FedAvg(0.01, 2)
    fedavg_easy.train(train_data, train_label, val_data, val_label, "easy", 200, 501)

    fedavg_m_easy = FedAvg(0.01, 2)
    fedavg_m_easy.train(train_data, train_label, val_data, val_label, "most_easy", 200, 501)

    fedavg_hard = FedAvg(0.01, 2)
    fedavg_hard.train(train_data, train_label, val_data, val_label, "hard", 300, 501)

    print("sgd start =========")
    sgd = SGD()
    sgd.train(train_data, train_label, val_data, val_label)

    print("fedsgd start =========")
    fedsgd_hard = FedSGD()
    fedsgd_hard.train(train_data, train_label, val_data, val_label, "hard")

    fedsgd_easy = FedSGD()
    fedsgd_easy.train(train_data, train_label, val_data, val_label, "easy", iter=701)


    # plot of communication round
    line_fedavg_easy, = plt.plot(fedavg_easy.traj, label='fedavg (easy)')
    line_fedavg_m_easy, = plt.plot(fedavg_m_easy.traj, label='fedavg (easist)')
    line_fedavg_hard, = plt.plot(fedavg_hard.traj, label='fedavg (hard)')
    line_fedavg_rd, = plt.plot(fedavg_rd.traj, label='fedavg (random)')

    # line_fedsgd_easy, = plt.plot(fedsgd_easy.traj, label='fedsgd (easy)')
    # line_fedsgd_hard, = plt.plot(fedsgd_hard.traj, label='fedsgd (hard)')

    # line_sgd, = plt.plot(sgd.traj, label='sgd')

    plt.ylabel("loss")
    plt.xlabel("communication round")
    plt.legend(handles=[line_fedavg_rd, line_fedavg_m_easy, line_fedavg_easy, line_fedavg_hard])
    # plt.legend(handles=[line_sgd, line_fedavg_rd, line_fedavg_easy, line_fedavg_hard, line_fedsgd_easy, line_fedsgd_hard])
    plt.title("val loss for each method")
    plt.show()

    # plot of time
    fedavg_easy_time, = plt.plot(fedavg_easy.time_traj, fedavg_easy.traj, label='fedavg (easy)')
    fedavg_m_easy_time, = plt.plot(fedavg_m_easy.time_traj, fedavg_m_easy.traj, label='fedavg (easist)')
    fedavg_hard_time, = plt.plot(fedavg_hard.time_traj, fedavg_hard.traj, label='fedavg (hard)')
    fedavg_rd_time, = plt.plot(fedavg_rd.time_traj, fedavg_rd.traj, label='fedavg (random)')

    # fedsgd_easy_time, = plt.plot(fedsgd_easy.time_traj, fedsgd_easy.traj, label='fedsgd (easy)')
    # fedsgd_hard_time, = plt.plot(fedsgd_easy.time_traj, fedsgd_hard.traj, label='fedsgd (hard)')

    # sgd_time, = plt.plot(sgd.time_traj, sgd.traj, label='sgd')

    plt.ylabel("loss")
    plt.xlabel("time")
    plt.legend(handles=[fedavg_rd_time, fedavg_m_easy_time, fedavg_easy_time, fedavg_hard_time])
    # plt.legend(handles=[sgd_time, fedavg_rd_time, fedavg_easy_time, fedavg_hard_time, fedsgd_easy_time, fedsgd_hard_time])
    plt.title("val loss against time for each method")
    plt.show()

    # plot of number of gradient descent
    fedavg_easy_n, = plt.plot(fedavg_easy.update_num, fedavg_easy.traj, label='fedavg (easy)')
    fedavg_m_easy_n, = plt.plot(fedavg_m_easy.update_num, fedavg_m_easy.traj, label='fedavg (easist)')
    fedavg_hard_n, = plt.plot(fedavg_hard.update_num, fedavg_hard.traj, label='fedavg (hard)')
    fedavg_rd_n, = plt.plot(fedavg_rd.update_num, fedavg_rd.traj, label='fedavg (random)')

    # fedsgd_easy_n, = plt.plot(fedsgd_easy.update_num, fedsgd_easy.traj, label='fedsgd (easy)')
    # fedsgd_hard_n, = plt.plot(fedsgd_easy.update_num, fedsgd_hard.traj, label='fedsgd (hard)')

    # sgd_n, = plt.plot(sgd.update_num, sgd.traj, label='sgd')

    plt.ylabel("loss")
    plt.xlabel("number of updates")
    plt.legend(handles=[fedavg_rd_n, fedavg_m_easy_n, fedavg_easy_n, fedavg_hard_n])
    # plt.legend(handles=[sgd_n, fedavg_rd_n, fedavg_easy_n, fedavg_hard_n, fedsgd_easy_n, fedsgd_hard_n])
    plt.title("val loss against number of gradient descents for each method")
    plt.show()
