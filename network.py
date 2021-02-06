import autograd.numpy as np
import autograd as ag

DIM = 100


def random_init(std, nhid):
    return {'W1': np.random.normal(0, std, size=(DIM, nhid)),
            'b1': np.random.normal(0., std, size=nhid),
            'W2': np.random.normal(0., std, size=(nhid, nhid)),
            'b2': np.random.normal(0., std, size=nhid),
            'W3': np.random.normal(0., std, size=(nhid, nhid)),
            'b3': np.random.normal(0., std, size=nhid),
            'W4': np.random.normal(0., std, size=nhid),
            'b4': np.random.normal(0., std)}


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def relu(z):
    return np.maximum(z, 0.)


def net_predict(params, x):
    """Compute the output of a ReLU MLP with 2 hidden layers."""
    H1 = relu(np.dot(x, params['W1']) + params['b1'])
    H2 = relu(np.dot(H1, params['W2']) + params['b2'])
    H3 = relu(np.dot(H2, params['W3']) + params['b3'])
    res = np.dot(H3, params['W4']) + params['b4']
    return res


def gd_step(cost, params, lrate):
    """Perform one gradient descent step on the given cost function with learning
    rate lrate. Returns a new set of parameters, and (IMPORTANT) does not modify
    the input parameters."""
    new_params = {}
    grad_cost = ag.grad(cost)
    gradient = grad_cost(params)
    for p in params:
        new_params[p] = params[p] - lrate * gradient[p]
    return new_params


def sample_batch(train_data, train_label, batch_size):
    sample = list(np.random.permutation((train_data.shape[0])))[0: batch_size]
    batch_data = np.asarray([train_data[i] for i in sample])
    batch_label = np.asarray([train_label[i] for i in sample])
    return batch_data, batch_label
