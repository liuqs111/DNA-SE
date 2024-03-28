import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
import time
import math

'''
0. utilize GPU 1
torch.cuda.is_available() 
torch.cuda.device_count() : Returns the number of GPUs available
'''
torch.set_default_dtype(torch.float64)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

'''
1. generate data
'''


def expit(x):
    return torch.sigmoid(x)

N = 10000 # sample size
m = 1000 # monte carlo size
B = 5000
w = 9  # width = w*(1+p)
BATCHSIZE = 5000
EPOCH = 20000


def weight_init(w):
    if isinstance(w, nn.Linear):
        nn.init.xavier_normal_(w.weight)

    elif isinstance(w, nn.Conv2d):
        nn.init.kaiming_normal_(w.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(w, nn.BatchNorm2d):
        nn.init.constant_(w.weight, 1)
        nn.init.constant_(w.bias, 0)

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
print(50 * '=')

st = time.time()

tensor = []

for item in range(0, 50):


    class g_func(nn.Module):
        def __init__(self):
            super(g_func, self).__init__()
            self.fc1 = nn.Linear(1, 5)
            self.fc2 = nn.Linear(5, 1)
            self.nn_ac_fun = nn.Sigmoid()

            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)

        def forward(self, x):
            x = self.nn_ac_fun(self.fc1(x))
            x = self.fc2(x)
            return x


    np.random.seed(item)
    torch.manual_seed(item)
    data_train0 = 2 * torch.rand(B, 1) + 1
    train_data = Data.TensorDataset(data_train0, torch.zeros(B, 1))
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)

    g_net = g_func().to(device)

    Parameters_g = [{"params": g_net.parameters()}]
    optimizer_g = torch.optim.Adam(Parameters_g, lr=1e-4)
    lr_beta_init = optimizer_g.param_groups[0]['lr']
    
    np.random.seed(item)
    torch.manual_seed(item)

    for epoch in range(EPOCH):
        for step, (data_train, _) in enumerate(train_loader):
            def f(x):
                fx = expit(x**(1/2))

                return fx


            def F_func(a, x, y, func1, func2):
                temp1 = y - expit(x + 1)

                temp2 = y - expit(x)

                return a * temp1 * func1(x) + (1 - a) * temp2 * func2(x)


            def v_func(x, a):
                y_1 = a * expit(x + 1) + (1 - a) * expit(x)

                y_ = y_1 - y_1 ** 2

                a_given_x = a * expit(x) + (1 - a) * (1 - expit(x))

                return y_ * a_given_x


            def l_func(x, y):
                return (y - f(x)) ** 2


            def rou(a):
                return 1 - torch.mean(a)


            def k_func(x):
                coef = v_func(x, 0) * torch.mean(v_func(x, 1)) / (v_func(x, 1) * 1 ** 2)

                loss_1 = (l_func(x, 1) - l_func(x, 0))

                return coef * loss_1


            def mu_func(x):
                left = torch.mean(v_func(x, 1))

                right = 1 + v_func(x, 0) / (v_func(x, 1) * (1 ** 2))

                return left * right
            

            data_train = data_train.to(device)

            data_train_clone = data_train.clone()  # n by p
            pa = expit(data_train_clone)
            a = torch.bernoulli(pa).to(device)

            rou1 = 1 - torch.mean(a)

            Coef = torch.mean(g_net(data_train_clone) * v_func(data_train_clone, 1) / mu_func(data_train_clone))
            sum_left = Coef + k_func(data_train_clone) / rou1
            
            sum_right = g_net(data_train_clone)

            loss_g = torch.mean((sum_left - sum_right) ** 2)
            print(loss_g)

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()


            def g1_func(x):
                return -torch.mean(g_net(x) * v_func(x, 1) / mu_func(x)) * 1 / (torch.mean(v_func(x, 1))) +1 * g_net(x) / mu_func(x)
            
            
            def g2_func(x):

                return g_net(x) / mu_func(x)


            X = 2 * torch.rand(B,1) + 1

            X = X.to(device)

            pA = expit(X) # \theta = x
            A = torch.bernoulli(pA).to(device)
            # calculate P(Y=1|X=x, A=0)
            p_Y_A0 = expit(X)

            # calculate P(Y=1|X=x, A=1)
            p_Y_A1 = expit(X+1)

            py = A * expit(X+1) + (1 - A) * expit(X)  # \phi = x+1
            Y = torch.bernoulli(py).to(device)
            
            
            X2 = 2 * torch.rand(100000,1) + 1

            X2 = X2.to(device)

            pA2 = expit(X2) # \theta = x
            A2 = torch.bernoulli(pA2).to(device)
            # calculate P(Y=1|X=x, A=0)
            p_Y2_A0 = expit(X2)

            # calculate P(Y=1|X=x, A=1)
            p_Y2_A1 = expit(X2+1)

            py2 = A2 * expit(X2+1) + (1 - A2) * expit(X2)  # \phi = x+1
            Y2 = torch.bernoulli(py2).to(device)

            A1 = A.clone()
            X1 = X.clone()
            Y1 = Y.clone()

            F = F_func(A1, X1, Y1, g1_func, g2_func)

            r_star = torch.mean((Y1 - f(X1))**2)

            mask1 = A2 == 0

            Y2_masked = Y2[mask1].clone().reshape(-1, 1)
            X2_masked = X2[mask1].clone().reshape(-1, 1)
            A2_masked = A2[mask1].clone().reshape(-1, 1)
            mean_value1 = torch.mean((Y2_masked - f(X2_masked)) ** 2)

            D_cspd = (1 - A1) * ((Y1 - f(X1)) ** 2 - torch.mean(mean_value1)) / rou(A1) + F

            loss_theta = torch.mean( (D_cspd))
            
    et = time.time()

    print(item, ':', loss_g.data, loss_theta.data, et - st)
    tensor.append(loss_theta)
    torch.save(g_net.to(torch.device('cpu')), 'a' + str(item) + '.pkl')

file_path = "transfer_learning.txt"

with open(file_path, "w") as file:
    for i in tensor:
        tensor_str = str(i)
        file.write(tensor_str + "\n")




