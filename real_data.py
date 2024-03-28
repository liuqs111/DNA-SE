import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
import time

# from torch.utils.tensorboard import SummaryWriter

'''
0. utilize GPU 1
torch.cuda.is_available() 
torch.cuda.device_count() : Returns the number of GPUs available
'''

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

'''
1. generate data
'''

alternating_freq = 1
N1 = 2486  # sample size of shadow model
h = 3 * (2000 ** (-1 / 5))
p = 3
M = 1000  # MC sample size
B = 500  # training sample size
EPOCH = 15000  # training epochs
BATCHSIZE = 500  # minibatch training sample size
# total updating steps = EPOCH*(B/BATCHSIZE)
pi = torch.FloatTensor([np.pi]).cuda(2)


def pi_star(y, u):
    return torch.sigmoid(1.058 - 2.037 * y + 0.298 * u[:, [1]] - 0.002 * u[:, [0]])


def kernel(x):
    # return 3/4*(1-x**2)*(abs(x)<1)
    x = torch.sqrt(torch.sum(x ** 2, dim=1)).reshape(-1, 1)
    x = x / h
    return 45 / 32 * (1 - 7 / 3 * x ** 2) * (1 - x ** 2) * (abs(x) <= 1) / h


np.random.seed(9999)
torch.manual_seed(9999)

seeds = torch.randint(100000, size=(50, 1))

for item in range(1):
    for nodes in [15]:

        np.random.seed(seeds[item])
        torch.manual_seed(seeds[item])

        newdata = np.loadtxt('../newdata.txt')
        U_father = 1 - torch.Tensor(newdata[:, [0]].copy())
        U_health = torch.Tensor(newdata[:, [1]].copy())
        U1 = torch.hstack([U_father, U_health])
        Z1 = torch.Tensor(newdata[:, [4]].copy())
        R1 = torch.Tensor(newdata[:, [3]].copy())
        Y1 = torch.Tensor(newdata[:, [2]].copy())

        bootstrap_iterations = 50  
        N = bootstrap_sample_size = 2000  

        U = []
        Z = []
        R = []
        Y = []

        for _ in range(bootstrap_iterations):

            indices = np.random.choice(U1.shape[0], N, replace=False)
            U = U1[indices, :]
            Z = Z1[indices, :]
            R = R1[indices, :]
            Y = Y1[indices, :]

            U = U.cuda(2)
            Z = Z.cuda(2)
            Y = Y.cuda(2)
            R = R.cuda(2)

            '''
            2. define neural network 
            '''

            beta = nn.Linear(p + 1, 1, bias=False)
            b0 = nn.Sequential(
                nn.Linear(p, nodes),
                nn.Tanh(),
                nn.Linear(nodes, 1)
            )

            b1 = nn.Sequential(
                nn.Linear(p, nodes),
                nn.Tanh(),
                nn.Linear(nodes, 1)
            )

            b2 = nn.Sequential(
                nn.Linear(p, nodes),
                nn.Tanh(),
                nn.Linear(nodes, 1)
            )

            b3 = nn.Sequential(
                nn.Linear(p, nodes),
                nn.Tanh(),
                nn.Linear(nodes, 1)
            )

            beta.cuda(2)
            b0.cuda(2)
            b1.cuda(2)
            b2.cuda(2)
            b3.cuda(2)

            '''
            3. Training
            '''


            # initialize the weight
            def weight_init(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)

                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


            b0.apply(weight_init)
            b1.apply(weight_init)
            b2.apply(weight_init)
            b3.apply(weight_init)
            beta.apply(weight_init)

            # generate training data and set mini batch
            # y_train0 = torch.linspace(int(min(Y)) - 0.5, int(max(Y)) + 0.5, B).reshape(B, 1)
            yu_train0 = torch.hstack([torch.rand(B, 1), torch.rand(B, 1), torch.rand(B, 1)])
            train_data = Data.TensorDataset(yu_train0, torch.zeros(B, 1))
            train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)

            # set parameters and optimizer
            Parameters_b = [{"params": b0.parameters()}, {"params": b1.parameters()}, {"params": b2.parameters()},
                            {"params": b3.parameters()}]
            optimizer_b = torch.optim.Adam(Parameters_b, lr=1e-2)
            Parameters_beta = beta.parameters()
            optimizer_beta = torch.optim.Adam(Parameters_beta, lr=1e-2)

            binary = torch.ones(2, 1).cuda(2)
            binary[0] = 0
            ones_N = torch.ones(N, 1).cuda(2)
            ones_2 = torch.ones(2, 1).cuda(2)
            ones_2N = torch.ones(2 * N, 1).cuda(2)
            b = BATCHSIZE
            ones_b = torch.ones(b, 1).cuda(2)
            st = time.time()

            for epoch in range(EPOCH):
                for step, (yu_train, _) in enumerate(train_loader):
                    yu_train = yu_train.cuda(2)

                    y_train = yu_train[:, [0]]
                    u_train = yu_train[:, 1:3].clone()


                    def p_y(y, u, z):
                        n_dim = y.shape[0]
                        py = torch.sigmoid(beta(torch.hstack([torch.ones(n_dim, 1).cuda(2), u, z])))
                        return y * py + (1 - y) * (1 - py)


                    def p_beta0(y, u, z):
                        n_dim = y.shape[0]
                        ey = torch.exp(beta(torch.hstack([torch.ones(n_dim, 1).cuda(2), u, z])))
                        return (2 * y - 1) * ey / (1 + ey) ** 2


                    def p_beta1(y, u, z):
                        n_dim = y.shape[0]
                        ey = torch.exp(beta(torch.hstack([torch.ones(n_dim, 1).cuda(2), u, z])))
                        return u[:, [0]] * (2 * y - 1) * ey / (1 + ey) ** 2


                    def p_beta2(y, u, z):
                        n_dim = y.shape[0]
                        ey = torch.exp(beta(torch.hstack([torch.ones(n_dim, 1).cuda(2), u, z])))
                        return u[:, [1]] * (2 * y - 1) * ey / (1 + ey) ** 2


                    def p_beta3(y, u, z):
                        n_dim = y.shape[0]
                        ey = torch.exp(beta(torch.hstack([torch.ones(n_dim, 1).cuda(2), u, z])))
                        return z * (2 * y - 1) * ey / (1 + ey) ** 2


                    yu_train_ber = torch.bernoulli(yu_train)
                    y_train_ber = yu_train_ber[:, [0]]
                    u_train_ber = yu_train_ber[:, 1:3].clone()
                    y_Nb = torch.kron(y_train_ber, ones_N)
                    u_Nb = torch.kron(u_train_ber, ones_N)
                    Z_Nb = torch.kron(ones_b, Z)
                    uU_Nb = torch.kron(ones_b, U) - torch.kron(u_train_ber, ones_N)

                    h_yu = torch.mean((p_y(y_Nb, u_Nb, Z_Nb).data * kernel(uU_Nb)).reshape(b, N), dim=1).reshape(-1, 1)
                    b0_yu = b0(yu_train_ber)
                    b1_yu = b1(yu_train_ber)
                    b2_yu = b2(yu_train_ber)
                    b3_yu = b3(yu_train_ber)

                    u_2Nb = torch.kron(u_train_ber, ones_2N)
                    zz = torch.kron(Z, ones_2)
                    Z_2Nb = torch.kron(ones_b, zz)
                    binary_2Nb = torch.kron(ones_b, torch.kron(ones_N, binary))
                    tu_2Nb = torch.hstack([binary_2Nb, u_2Nb])

                    int_denom = 1 - torch.sum(
                        (p_y(binary_2Nb, u_2Nb, Z_2Nb).data * pi_star(binary_2Nb, u_2Nb)).reshape(N * b, 2),
                        dim=1).reshape(-1, 1)
                    int_num0 = torch.sum(
                        (b0(tu_2Nb) * pi_star(binary_2Nb, u_2Nb) * p_y(binary_2Nb, u_2Nb, Z_2Nb).data).reshape(N * b,
                                                                                                               2),
                        dim=1).reshape(-1, 1)
                    int_num1 = torch.sum(
                        (b1(tu_2Nb) * pi_star(binary_2Nb, u_2Nb) * p_y(binary_2Nb, u_2Nb, Z_2Nb).data).reshape(N * b,
                                                                                                               2),
                        dim=1).reshape(-1, 1)
                    int_num2 = torch.sum(
                        (b2(tu_2Nb) * pi_star(binary_2Nb, u_2Nb) * p_y(binary_2Nb, u_2Nb, Z_2Nb).data).reshape(N * b,
                                                                                                               2),
                        dim=1).reshape(-1, 1)

                    int_num3 = torch.sum(
                        (b3(tu_2Nb) * pi_star(binary_2Nb, u_2Nb) * p_y(binary_2Nb, u_2Nb, Z_2Nb).data).reshape(N * b,
                                                                                                               2),
                        dim=1).reshape(-1, 1)

                    f0_yu = torch.mean(
                        (p_y(y_Nb, u_Nb, Z_Nb).data * int_num0 * kernel(uU_Nb) / int_denom).reshape(b, N),
                        dim=1).reshape(-1, 1)
                    f1_yu = torch.mean(
                        (p_y(y_Nb, u_Nb, Z_Nb).data * int_num1 * kernel(uU_Nb) / int_denom).reshape(b, N),
                        dim=1).reshape(-1, 1)
                    f2_yu = torch.mean(
                        (p_y(y_Nb, u_Nb, Z_Nb).data * int_num2 * kernel(uU_Nb) / int_denom).reshape(b, N),
                        dim=1).reshape(-1, 1)
                    f3_yu = torch.mean(
                        (p_y(y_Nb, u_Nb, Z_Nb).data * int_num3 * kernel(uU_Nb) / int_denom).reshape(b, N),
                        dim=1).reshape(-1, 1)

                    int_num0_g = torch.sum(
                        (p_beta0(binary_2Nb, u_2Nb, Z_2Nb).data * pi_star(binary_2Nb, u_2Nb)).reshape(N * b, 2),
                        dim=1).reshape(-1, 1)
                    int_num1_g = torch.sum(
                        (p_beta1(binary_2Nb, u_2Nb, Z_2Nb).data * pi_star(binary_2Nb, u_2Nb)).reshape(N * b, 2),
                        dim=1).reshape(-1, 1)
                    int_num2_g = torch.sum(
                        (p_beta2(binary_2Nb, u_2Nb, Z_2Nb).data * pi_star(binary_2Nb, u_2Nb)).reshape(N * b, 2),
                        dim=1).reshape(-1, 1)
                    int_num3_g = torch.sum(
                        (p_beta3(binary_2Nb, u_2Nb, Z_2Nb).data * pi_star(binary_2Nb, u_2Nb)).reshape(N * b, 2),
                        dim=1).reshape(-1, 1)

                    g0_yu = torch.mean(((p_beta0(y_Nb, u_Nb, Z_Nb).data / p_y(y_Nb, u_Nb, Z_Nb).data +
                                         int_num0_g / int_denom) * p_y(y_Nb, u_Nb, Z_Nb) * kernel(uU_Nb)).reshape(b, N),
                                       dim=1).reshape(-1, 1)
                    g1_yu = torch.mean(((p_beta1(y_Nb, u_Nb, Z_Nb).data / p_y(y_Nb, u_Nb, Z_Nb).data +
                                         int_num1_g / int_denom) * p_y(y_Nb, u_Nb, Z_Nb) * kernel(uU_Nb)).reshape(b, N),
                                       dim=1).reshape(-1, 1)
                    g2_yu = torch.mean(((p_beta2(y_Nb, u_Nb, Z_Nb).data / p_y(y_Nb, u_Nb, Z_Nb).data +
                                         int_num2_g / int_denom) * p_y(y_Nb, u_Nb, Z_Nb) * kernel(uU_Nb)).reshape(b, N),
                                       dim=1).reshape(-1, 1)
                    g3_yu = torch.mean(((p_beta3(y_Nb, u_Nb, Z_Nb).data / p_y(y_Nb, u_Nb, Z_Nb).data +
                                         int_num3_g / int_denom) * p_y(y_Nb, u_Nb, Z_Nb) * kernel(uU_Nb)).reshape(b, N),
                                       dim=1).reshape(-1, 1)

                    res0 = torch.mean((h_yu * b0_yu + f0_yu - g0_yu) ** 2)
                    res1 = torch.mean((h_yu * b1_yu + f1_yu - g1_yu) ** 2)
                    res2 = torch.mean((h_yu * b2_yu + f2_yu - g2_yu) ** 2)
                    res3 = torch.mean((h_yu * b3_yu + f3_yu - g3_yu) ** 2)
                    loss_b = res0 + res1 + res2 + res3

                    print(loss_b)

                    optimizer_b.zero_grad()
                    loss_b.backward()
                    optimizer_b.step()

                if epoch % alternating_freq == 0:
                    U_2N = torch.kron(U, ones_2)
                    Z_2N = torch.kron(Z, ones_2)
                    binary_2N = torch.kron(ones_N, binary)

                    int_denom_S = 1 - torch.mean((p_y(binary_2N, U_2N, Z_2N) * pi_star(binary_2N, U_2N)).reshape(N, 2),
                                                 dim=1).reshape(-1, 1)

                    int_b0_S = torch.mean(
                        ((b0(torch.hstack([binary_2N, U_2N])).data * p_y(binary_2N, U_2N, Z_2N) -
                          p_beta0(binary_2N, U_2N, Z_2N)) * pi_star(binary_2N, U_2N)).reshape(N, 2),
                        dim=1).reshape(-1, 1)
                    int_b1_S = torch.mean(
                        ((b1(torch.hstack([binary_2N, U_2N])).data * p_y(binary_2N, U_2N, Z_2N) -
                          p_beta1(binary_2N, U_2N, Z_2N)) * pi_star(binary_2N, U_2N)).reshape(N, 2),
                        dim=1).reshape(-1, 1)
                    int_b2_S = torch.mean(
                        ((b2(torch.hstack([binary_2N, U_2N])).data * p_y(binary_2N, U_2N, Z_2N) -
                          p_beta2(binary_2N, U_2N, Z_2N)) * pi_star(binary_2N, U_2N)).reshape(N, 2),
                        dim=1).reshape(-1, 1)
                    int_b3_S = torch.mean(
                        ((b3(torch.hstack([binary_2N, U_2N])).data * p_y(binary_2N, U_2N, Z_2N) -
                          p_beta3(binary_2N, U_2N, Z_2N)) * pi_star(binary_2N, U_2N)).reshape(N, 2),
                        dim=1).reshape(-1, 1)

                    YU = torch.hstack([Y, U])
                    b0_YU = b0(YU)
                    b1_YU = b1(YU)
                    b2_YU = b2(YU)
                    b3_YU = b3(YU)

                    S_eff0 = torch.mean(
                        R * (p_beta0(Y, U, Z) / p_y(Y, U, Z) - b0_YU) + (1 - R) * int_b0_S / int_denom_S)

                    S_eff1 = torch.mean(
                        R * (p_beta1(Y, U, Z) / p_y(Y, U, Z) - b1_YU) + (1 - R) * int_b1_S / int_denom_S)

                    S_eff2 = torch.mean(
                        R * (p_beta2(Y, U, Z) / p_y(Y, U, Z) - b2_YU) + (1 - R) * int_b2_S / int_denom_S)

                    S_eff3 = torch.mean(
                        R * (p_beta3(Y, U, Z) / p_y(Y, U, Z) - b3_YU) + (1 - R) * int_b3_S / int_denom_S)

                    loss_beta = S_eff0 ** 2 + S_eff1 ** 2 + S_eff2 ** 2 + S_eff3 ** 2

                    print(loss_beta, 80 * "=")

                    optimizer_beta.zero_grad()
                    loss_beta.backward()
                    optimizer_beta.step()

            print(item, nodes, beta.state_dict().items())

            # save data
            directory = "beta_files/"
            for key, value in beta.state_dict().items():
                file_path = directory + str(key) + "3.txt"
                with open(file_path, "a") as file:
                    file.write(str(value) + '\n')
            et = time.time()
            print(et - st)


