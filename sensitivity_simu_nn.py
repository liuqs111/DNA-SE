import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
import time
from torch.utils.tensorboard import SummaryWriter

'''
0. utilize GPU 1
torch.cuda.is_available() 
torch.cuda.device_count() : Returns the number of GPUs available
'''
torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
'''
1. generate data
'''


def expit(x):
    return torch.sigmoid(x)


np.random.seed(9)
torch.manual_seed(9)

seeds = torch.randint(100000, size = (50,1))

N = 2000
p = 10
m = 100+1
B = 5000
w = 9  # width = w*(1+p)
BATCHSIZE = 5000
EPOCH = 20000
sample_type = 'random'  # 'random' 'partition_perm' 'fixed_X'
ac_fun = 'ReLU'
n = BATCHSIZE
alter_freq = 1
alpha =0.01  # penalty parameter
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
#kappa[p-1] = 3
#llambda[p-1] = 4
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

def weight_init_beta(w):
    if isinstance(w, nn.Linear):
        nn.init.constant_(w.weight, 0)


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
print('v_sample = ',v_sample)
#print('network: ', a)
print(50 * '=')

st = time.time()

for item in range(5,30):

    for w,l in [[w,l] for w in [3] for l in [5]]:
        
        #np.random.seed(seeds[item])
        #torch.manual_seed(seeds[item])
        
        np.random.seed(item)
        torch.manual_seed(item)
        
        X = torch.rand(N, p)
        # U = torch.randn(N,1) * 0.1 + X[:,[0]] - X[:,[1]]**2
        
        X = X.to(device)
        U = U.to(device)
        U = torch.distributions.Beta(2, 2).sample([N, 1]).to(device)
        U = U.to(device)
        pz = expit(X @ kappa + gamma * U)
        Z = torch.bernoulli(pz).to(device)
        py = expit(X @ llambda + beta * Z + delta * U)
        Y = torch.bernoulli(py).to(device)
    
        np.random.seed(item)
        torch.manual_seed(item)    
        if v_sample == 'mesh':
            m = 100 + 1
            v = torch.linspace(0,1,m).reshape(m,1).to(device)
            h = 1 / (m - 1)
            W = (h*torch.diag(torch.hstack([0.5*torch.ones(1), torch.ones(m - 2), 0.5*torch.ones(1)]))).to(device)
        elif v_sample == 'mc':
            m = 100 + 1
            v = torch.rand(m,1) - 0.5
            v = v.to(device)
            #v = torch.rand(m, 1).to(device)

        np.random.seed(item)
        torch.manual_seed(item)
        data_train0 = training_sample(sample_type)
        data_train0[:,[0]] = torch.rand(B,1) - 0.5
        train_data = Data.TensorDataset(data_train0, torch.zeros(B, 1))
        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)
    
        beta_net = nn.Linear(1, 1, bias=False)
        a = nn.Sequential(
            nn.Linear(1 + p, w * (1 + p)),
            nn_ac_fun,
            *[ nn.Linear(w * (1 + p), w * (1 + p)) if i%2==0 else nn.ReLU() for i in range(l*2)],
            nn.Linear(w * (1 + p), 1)
        )

        beta_net = beta_net.to(device)
        a = a.to(device)
    
        a.apply(weight_init)
        beta_net.apply(weight_init_beta)
    
        Parameters_a = [{"params": a.parameters()}]
        optimizer_a = torch.optim.Adam(Parameters_a, lr=1e-4)
 
        Parameters_beta = beta_net.parameters()
        optimizer_beta = torch.optim.RMSprop(Parameters_beta)
        lr_beta_init = optimizer_beta.param_groups[0]['lr']
    
        writer = SummaryWriter()
        for epoch in range(EPOCH):
            for step, (data_train, _) in enumerate(train_loader):

                adjust_learning_rate(optimizer_beta, epoch, lr_beta_init)
                lr = optimizer_beta.param_groups[0]['lr']


                def fyz_vx(yzvx):
                    y = yzvx[:, [0]]
                    z = yzvx[:, [1]]
                    v = yzvx[:, [2]]
                    x = yzvx[:, 3:(3 + p)].clone()

                    pz = expit(x @ kappa + gamma * v)
                    py = expit(x @ llambda + beta_net(z) + delta * v)
                    return (z * pz + (1 - z) * (1 - pz)) * (y * py + (1 - y) * (1 - py))


                def S(yzvx):
                    y = yzvx[:, [0]]
                    z = yzvx[:, [1]]
                    v = yzvx[:, [2]]
                    x = yzvx[:, 3:(3 + p)].clone()
 
                    py = expit(x @ llambda + beta_net(z) + delta * v)
                    return (y - py) * z


                data_train = data_train.to(device)
                x = data_train[:, 1:(p + 1)].clone()  # n by p
            
                yz = torch.Tensor([[y, z] for y in range(2) for z in range(2)]).to(device)  # 4 by 2
                vx = torch.hstack([torch.kron(ones_n, v), torch.kron(x, ones_m)])
                yzvx = torch.hstack([torch.kron(yz, ones_mn), torch.kron(ones_4, vx)])

                p_yzvx = fyz_vx(yzvx).data.to(device)  # mn4 by 1
                s_yzvx = S(yzvx).data.to(device)  # mn4 by 1
                a_vx = torch.kron(ones_4, a(vx))  # 4mn by 1
            
                if v_sample == 'mesh':
                    num = torch.sum((p_yzvx * (a_vx - s_yzvx)).reshape(4, n, m)@W, dim=2)  # 4 by n
                    denom = torch.sum(p_yzvx.reshape(4, n, m)@W, dim=2)  # 4 by n
                elif v_sample == 'mc':        
                    num = torch.mean((p_yzvx * (a_vx - s_yzvx)).reshape(4, n, m), dim=2)  # 4 by n
                    denom = torch.mean(p_yzvx.reshape(4, n, m), dim=2)  # 4 by n
            # reshape and sum over m
                #num = torch.sum((p_yzvx * (a_vx - s_yzvx)).reshape(4, n, m)@W, dim=2)  # 4 by n
                #denom = torch.sum(p_yzvx.reshape(4, n, m)@W, dim=2)  # 4 by n
            
            
                S_eff = num / (denom + 1e-20*(denom==0))  # 4 by n

            # sum over yz
                yzux = torch.hstack([torch.kron(yz, ones_n), torch.kron(ones_4, data_train)])  # 4n by 1
                p_yzux = fyz_vx(yzux).reshape(4, n)  # 4 by n

                sum_yz = torch.sum(S_eff * p_yzux, dim=0).reshape(n, 1)  # n by 1
                a_ux = a(data_train)  # n by 1
                loss_a = torch.mean((sum_yz - alpha * a_ux) ** 2)

                optimizer_a.zero_grad()
                loss_a.backward()
                optimizer_a.step()

                if epoch % alter_freq == 0:
                
                    vX = torch.hstack([torch.kron(ones_N, v), torch.kron(X, ones_m)])  # mN by 3
                    a_vX = a(vX).data.to(device)  # mN by 1
                    YZvX = torch.hstack([torch.kron(Y, ones_m), torch.kron(Z, ones_m), vX])  # mN by 5

                    s_YZvX = S(YZvX)  # mN by 1
                    p_YZvX = fyz_vx(YZvX)  # mN by 1
                    
                    if v_sample == 'mesh':
                        num = torch.sum(((s_YZvX - a_vX) * p_YZvX).reshape(N, m)@W, dim=1).reshape(N, 1)  # N
                        denom = torch.sum(p_YZvX.reshape(N, m)@W, dim=1).reshape(N, 1)  # N
                    elif v_sample == 'mc':        
                        num = torch.mean(((s_YZvX - a_vX) * p_YZvX).reshape(N, m), dim=1).reshape(N, 1)  # N
                        denom = torch.mean(p_YZvX.reshape(N, m), dim=1).reshape(N, 1)  # N
                
                    #num = torch.sum(((s_YZvX - a_vX) * p_YZvX).reshape(N, m)@W, dim=1).reshape(N, 1)  # N
                    #denom = torch.sum(p_YZvX.reshape(N, m)@W, dim=1).reshape(N, 1)  # N

                    S_eff_vec = num / (denom + 1e-20*(denom==0))

                    loss_beta = torch.mean(S_eff_vec) ** 2

                    optimizer_beta.zero_grad()
                    loss_beta.backward()
                    optimizer_beta.step()
                
                if epoch % 50 == 0:
                    writer.add_scalar('loss_a', loss_a, global_step=epoch)
                    writer.add_scalar('loss_beta', loss_beta, global_step=epoch)
                    writer.add_scalar('beta', beta_net.weight, global_step=epoch)

                if epoch == 100 and step == 50:
                    last_b0_simu = a(data_train).to('cpu').data.clone()
                    last_b0_real = ((g0(y) - f0(y)) / h(y)).to('cpu').data.clone()
                    x_line = y_train.to('cpu').data.clone()
                    plt.plot(x_line, last_b0_real, color='blue', label='Line')
                    plt.plot(x_line, last_b0_simu, color='red', label='Line')

                    plt.legend()
                    plt.xlabel('X')
                    plt.ylabel('Y')

                    plt.show()
                    plt.savefig('C:/Users/pku17/Desktop/game/plot.pdf')
        et = time.time()

        print(item,':',seeds[item], w, l,loss_a.data, loss_beta.data, beta_net.weight.item(),et - st)
        torch.save(a.to(torch.device('cpu')), 'a'+str(item)+'.pkl')


    


