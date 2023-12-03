import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
import time
import math
from torch.utils.tensorboard import SummaryWriter

'''
0. utilize GPU 1
torch.cuda.is_available() 
torch.cuda.device_count() : Returns the number of GPUs available
'''
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
'''
1. generate data
'''


def expit(x):
    return torch.sigmoid(x)


np.random.seed(9)
torch.manual_seed(9)

seeds = torch.randint(100000, size=(50, 1))

N = 2000
p = 10
m = 100 + 1
B = 5000
w = 9  # width = w*(1+p)
BATCHSIZE = 5000
EPOCH = 20000
sample_type = 'random'  # 'random' 'partition_perm' 'fixed_X'
ac_fun = 'ReLU'
n = BATCHSIZE
alter_freq = 1
alpha = 0.01  # penalty parameter
v_sample = 'mc'

if sample_type == 'fixed_X':
    k = 100
    ones_k = torch.ones(k, 1).to(device)
    B = N * k
    BATCHSIZE = N
    n = BATCHSIZE

ones_4 = torch.ones(4, 1).to(device)
ones_m = torch.ones(m, 1).to(device)
ones_n = torch.ones(n, 1).to(device)
ones_N = torch.ones(N, 1).to(device)
ones_mn = torch.ones(m * n, 1).to(device)

kappa = 3 * torch.Tensor([(-1) ** i for i in range(p)]).reshape(p, 1).to(device)
llambda = 4 * torch.Tensor([(-1) ** i for i in range(p)]).reshape(p, 1).to(device)
# kappa[p-1] = 3
# llambda[p-1] = 4
gamma = 2  # torch.FloatTensor([2]).to(device)
delta = 2  # torch.FloatTensor([2]).to(device)
beta = 2  # torch.FloatTensor([2]).to(device)

d = 4

if ac_fun == 'ReLU':
    nn_ac_fun = nn.ReLU()
else:
    nn_ac_fun = nn.Tanh()


def weight_init(w):
    if isinstance(w, nn.Linear):
        nn.init.xavier_normal_(w.weight)

    elif isinstance(w, nn.Conv2d):
        nn.init.kaiming_normal_(w.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(w, nn.BatchNorm2d):
        nn.init.constant_(w.weight, 1)
        nn.init.constant_(w.bias, 0)


def training_sample(sample_type):
    if sample_type == 'random':
        return torch.hstack([torch.rand(B, 1) * 1.4 - 0.2 for _ in range(p + 1)])
    elif sample_type == 'partition_perm':
        partition = torch.linspace(-0.2, 1.2, B).reshape(B, 1)
        return torch.hstack([partition[index] for index in [torch.randperm(B) for _ in range(p + 1)]])
    elif sample_type == 'fixed_X':
        partition = torch.linspace(-0.2, 1.2, k).reshape(k, 1).to(device)
        return torch.hstack([torch.kron(ones_N, partition), torch.kron(X, ones_k)])


def adjust_learning_rate(optimizer, epoch, lr):
    if epoch <= 100:
        lr = 1e-2
    elif epoch <= 200:
        lr = 1e-2
    else:
        lr = 1e-2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


print(50 * '=')
print('Start!')
print('N = ', N)
print('m = ', m)
print('B = ', B)
print('batchsize = ', BATCHSIZE)
print('alpha = ', alpha)
print('alter_freq = ', alter_freq)
print('beta_lr = ', 'dynamic')
print('training data = ', sample_type)
print('v_sample = ', v_sample)
# print('network: ', a)
print(50 * '=')

st = time.time()

for item in range(0, 50):

    # class theta(nn.Module):
    #     def __init__(self):
    #         super(theta, self).__init__()
    #         self.fc1 = nn.Linear(1, 10)  # 输入维度为1，输出维度为10
    #         self.fc2 = nn.Linear(10, 1)  # 输入维度为10，输出维度为1
    #
    #     def forward(self, x):
    #         x = torch.relu(self.fc1(x))  # 使用ReLU作为激活函数
    #         x = self.fc2(x)
    #         return x


    class g_func(nn.Module):
        def __init__(self):
            super(g_func, self).__init__()
            self.fc1 = nn.Linear(1, 5)  # 输入维度为1，输出维度为5
            self.fc2 = nn.Linear(5, 1)  # 输入维度为5，输出维度为1
            self.nn_ac_fun = nn.Sigmoid()

            # 权重初始化
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)

        def forward(self, x):
            x = self.nn_ac_fun(self.fc1(x))  # 使用Tanh作为激活函数
            x = self.fc2(x)
            return x

    np.random.seed(item)
    torch.manual_seed(item)
    # data_train0 = training_sample(sample_type)
    data_train0 = 2 * torch.rand(B,1) + 1
    train_data = Data.TensorDataset(data_train0, torch.zeros(B, 1))
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)

    # theta_net = theta().to(device)
    g_net = g_func().to(device)

    # theta_net.apply(weight_init)
    # g_net.apply(weight_init)
    #
    # Parameters_theta = [{"params": theta_net.parameters()}]
    # optimizer_theta = torch.optim.Adam(Parameters_theta, lr=1e-8)

    Parameters_g = [{"params": g_net.parameters()}]
    optimizer_g = torch.optim.Adam(Parameters_g, lr=1e-4)
    lr_beta_init = optimizer_g.param_groups[0]['lr']



    np.random.seed(item)
    torch.manual_seed(item)
    if v_sample == 'mesh':
        m = 100 + 1
        v = torch.linspace(0, 1, m).reshape(m, 1).to(device)
        h = 1 / (m - 1)
        W = (h * torch.diag(torch.hstack([0.5 * torch.ones(1), torch.ones(m - 2), 0.5 * torch.ones(1)]))).to(device)
    elif v_sample == 'mc':
        m = 100 + 1
        v = torch.rand(m, 1) - 0.5
        v = v.to(device)
        # v = torch.rand(m, 1).to(device)

    writer = SummaryWriter()
    for epoch in range(EPOCH):
        for step, (data_train, _) in enumerate(train_loader):

            def f(x):
                fx = expit(x)

                return fx


            #
            # mask = A == 0
            #
            #
            # mean_value = torch.mean((Y[mask].clone().reshape(-1,1) - f(X[mask].clone().reshape(-1,1))) ** 2)

            def F_func(a, x, y, func1, func2):

                temp1 = y - expit(x ** (1/3))

                temp2 = y - expit(x)

                return a * temp1 * func1(x) + (1 - a) * temp2 * func2(x)


            def v_func(x, a):
                y_1 = a * expit(x ** (1/3)) + (1 - a) * expit(x)

                y_ = y_1 - y_1 ** 2

                a_given_x = a * expit(x) + (1 - a) * (1 - expit(x))

                return y_ * a_given_x


            def l_func(x, y):
                return (y - f(x)) ** 2


            def rou(a):
                return 1 - torch.mean(a)


            def k_func(x):
                coef = v_func(x, 0) * torch.mean(v_func(x, 1)) / (v_func(x, 1) * ((3 * x ** (-2/3)) ** 2))

                loss_1 = (l_func(x, 1) - l_func(x, 0))

                return coef * loss_1


            def mu_func(x):
                left = torch.mean(v_func(x, 1))

                right = 1 + v_func(x, 0) / (v_func(x, 1) * ((3 * x ** (-2/3)) ** 2))

                return left * right

            # adjust_learning_rate(optimizer_g, epoch, lr_beta_init)
            # lr = optimizer_g.param_groups[0]['lr']

            data_train = data_train.to(device)

            data_train_clone = data_train.clone()  # n by p
            pa = expit(data_train_clone)
            a = torch.bernoulli(pa).to(device)

            rou1 = 1 - torch.mean(a)

            Coef = torch.mean(g_net(data_train_clone) * v_func(data_train_clone, 1) / mu_func(data_train_clone))
            # loss_1 = (l_func(x, 1) - l_func(x, 0))
            # v1 = v_func(x, 1)
            # v2 = v_func(x, 0)

            sum_left = Coef + k_func(data_train_clone) / rou1
            # aaaaa = (3 * theta_net(x) ** 2) ** 2
            #
            sdfs= torch.min(v_func(data_train_clone, 1))
            aaaa = torch.max(torch.mean(v_func(data_train_clone, 1)) / (v_func(data_train_clone, 1) ))
            bbbb = torch.min(torch.mean(v_func(data_train_clone, 1)) / (v_func(data_train_clone, 1) ))

            sum_right = g_net(data_train_clone)

            loss_g = torch.mean((sum_left - sum_right) ** 2)

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()


            def g1_func(x):

                return -torch.mean(g_net(x) * v_func(x, 1) / mu_func(x)) * 3 * x ** 2 / (torch.mean(v_func(x, 1))) + 3 * x ** 2 * g_net(x) / mu_func(x)

            def g2_func(x):

                return g_net(x) / mu_func(x)


            X = 2 * torch.rand(B,1) + 1

            X = X.to(device)

            pA = expit(X)
            A = torch.bernoulli(pA).to(device)

            py = A * expit(X ** 3) + (1 - A) * expit(X)  # \phi = x^3
            Y = torch.bernoulli(py).to(device)

            A1 = A.clone()
            X1 = X.clone()
            Y1 = Y.clone()

            F = F_func(A1, X1, Y1, g1_func, g2_func)

            r_star = torch.mean((Y1 - f(X1)) ** 2)

            print(r_star)

            mask1 = A1 == 0

            Y1_masked = Y1[mask1].clone().reshape(-1, 1)
            X1_masked = X1[mask1].clone().reshape(-1, 1)
            mean_value1 = torch.mean((Y1_masked - f(X1_masked)) ** 2)

            # mean_value1 = torch.mean((Y1[mask1].clone().reshape(-1, 1) - f(X1[mask1].clone().reshape(-1, 1))) ** 2)

            D_cspd = (1 - A1) * (mean_value1 - r_star) / rou(A1) + F
            print(torch.mean(D_cspd))

            loss_theta = torch.mean(math.sqrt(N) * (D_cspd - r_star))

            #
            # optimizer_g.zero_grad()
            # loss_g.backward()
            # optimizer_g.step()

            # optimizer_theta.zero_grad()
            # print(80 * "+")
            # loss_theta.backward(retain_graph=True)
            # optimizer_theta.step()


        if epoch % 50 == 0:
            writer.add_scalar('loss_g', loss_g, global_step=epoch)
            # writer.add_scalar('loss_theta', loss_theta, global_step=epoch)
            # writer.add_scalar('theta', theta_net.weight, global_step=epoch)
    et = time.time()

    print(item, ':', seeds[item], loss_g.data, loss_theta.data, et - st)
    torch.save(g_net.to(torch.device('cpu')), 'a' + str(item) + '.pkl')





