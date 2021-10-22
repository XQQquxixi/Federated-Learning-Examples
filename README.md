
# Overview
The high level idea is to understand FedAvg algorithm from ***Communication-Efficient Learning of Deep Networks
from Decentralized Data*** with heterogeneous data. Non-iid data are usually harmful in the context of federated learning, the ultimate goal is to answer the questions: Are all kinds of non-iid partitions equally bad? If not, can we identify "good" or "bad" non-iid partitions? Can we improve FedAvg given some certain non-iid partition or clients' learning results. 

Our initial result is there indeed exists a difference between non-iid partitions, some make FedAvg converge faster, almost as fast as iid partition; Some make FedAvg more unstable. We observe this in a toy example (synthetic Gaussian). And our hypothesis (maybe trivial) is the difficulty of a partition is determined by how heterogeneous are the decision boundaries in each client.

# Motivating Examples
### MNIST CNN v.s. SHAKESPEARE LSTM
This is an example from the original FedAvg paper ***Communication-Efficient Learning of Deep Networks
from Decentralized Data***, in the non-iid column, although for both tasks there are speed up by using FedAvg, but the speed up for MNIST CNN (2.8X) is smaller than SHAKESPEARE LSTM (95.3X). What makes the difference between these two datasets? 

![alt text](https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/cnn%2Blstm.png)

### Synthetic Gaussian
A naive idea is when the data is more separated, then it is easier to classify. And if clients get easy to learn problem, FedAvg better off, we tested this idea below.
#### 2D, Convex Decision Boundary
4 Different Scenario: top two clusters positive, bottom two negative. Denote them long rectangle, small square, tall rectangle, big square respectively.

<p float="left">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/longr.png" width="230" height="230">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/smalls.png" width="230" height="230">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/tallr.png" width="230" height="230">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/bigs.png" width="230" height="230">
</p>

Partitions: Suppose 2 clients:
1. iid: random shuffle
2. hard: green + red; pink + blue
3. easy: green + blue; pink + red

When do we observe partition effect? The first two cases (long rectangle and smalls square). We will present results for weight divergence ***Federated Learning withNon-IID Data***, train accuracy, train loss, and test accuracy, test loss. For long rectangle scenario:

<p float="left">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/longr_wd.png" width="190" height="190">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/longr_train_acc.png" width="190" height="190">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/longr_train_loss.png" width="190" height="190">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/longr_test_acc.png" width="190" height="190">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/longr_test_loss.png" width="190" height="190">
</p>

For small square scenario:

<p float="left">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/smalls_wd.png" width="190" height="190">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/smalls_train_acc.png" width="190" height="190">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/smalls_train_loss.png" width="190" height="190">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/smalls_test_acc.png" width="190" height="190">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/smalls_test_loss.png" width="190" height="190">
</p>


While in the other two cases (long rectangle and big square), the partition effect disappear:
<p float="left">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/fedavg_wd_lr.png" width="350" height="300">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/fedavg_wd_bs.png" width="350" height="300">
</p>


#### Decision Boundary
From the smalls square and long rectangle experiments, apparently distance alone does not define hard or easy problems. Our new hypothesis is how heterogeneous  the decision boundaries are in each client defines hard v.s. easy. If this is true then changing local epoch E should make the learning different, concretely easy partition worse off (as clients overfit more), and hard partition better off (maybe). This is aligned with what we observed, but the effect is not significant (this is 12D):

<p float="left">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/E_easy.png" width="300" height="250">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/E_hard.png" width="300" height="250">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/E_iid.png" width="300" height="250">
</p>


### High Dimensional Gaussian, Non-convex Decision Boundary
Data Generating Procedure: For d-dim Gaussian, some k, \epsilon > 0, for each i \in [d], generate positive data centered at k(e_i-\epsilon), negative data centered at k(e_i+\epsilon). So there will be 2d clusters of Gaussian in total. E.g. 3d after PCA:

<img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/3d_partition.png" width="300" height="300">

Partitions: Suppose we have 4 clients.
1. iid: random shuffle
2. hard: each client gets positive negative data along the same e_i
3. easy: each client gets positive data from e_i, and negative data from e_{i+1}
4. easiest: each clients gets positive data from e_i, and negative data from e_{i+d/4}

12D results: 

<p float="left">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/12d_loss_round.png" width="300" height="300">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/12d_loss_time.png" width="300" height="300">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/12d_loss_updates.png" width="300" height="300">
</p>

Non-balanced number of labels: We also studied the effect of changing the ratio of positive and negative data. According to ***Federated Learning withNon-IID Data***, this should change the final accuracy, but by controlling the number of data in total while changing the ratio, the difference in final accuracy almost disappear. 

P.S. It might seem interesting that some curves go down then go up, actually they don't go up forever, they would go up and converge to some number, but it's hard to tell what does that mean.

<p float="left">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/wd_naive.png" width="350" height="300">
  <img src="https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/wd_ratio.png" width="350" height="300">
</p>

It is worthy noting that although weight divergence indeed has some positive relation to loss/accuracy, it is a very sensitive metric. For example, in previous plot, weight divergence for iid partition goes down and goes up, it changes wildly after around 200 iterations, when loss/accuracy have converged. In left plot, different ratios gives very different weight divergence within a partition, but actually their loss/accuracy doesn't really differ much. And changing the number of iteration/learning rate, while give similar loss/accuracy, give very different weight divergence.

### Real World Dataset: MNIST
We use a 3-layer ReLU network as client's model. There are 10 clients in total, E = 5, C = 0.25, batch size = 100.
Settings: 

1. iid: random shuffle
2. fedsgd: iid (random shuffle)
3. non iid:  each get two digits randomly
4. distinct: each get two visually distinct digits like 2 and 6
5. similar:  each get two visually similar digits like (1, 7), (3, 5), (8, 9)

Result: Accuracy against communication rounds, the difference between different non-iid partitions is not very obvious: 

![alt text](https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/mnist.png)

Increasing Num of Clients: closes up the difference between iid and non-iid partitions:

![alt text](https://github.com/XQQquxixi/Federated-Learning-Examples/blob/main/imgs/acc_num_clients.png)


