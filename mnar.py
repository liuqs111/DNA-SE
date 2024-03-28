import numpy as np
import torch
import torch.utils.data as Data
from torch import nn

import time

'''
1. generate data
'''


def _main_(N, p, M, B, EPOCH, BATCHSIZE):
    np.random.seed(9)
    torch.manual_seed(9)

    seeds = torch.randint(100000, size=(100, 1))
    for item in range(0, 100):

        # hyperparameters
        # sample size of original data
        N = N
        # dimension of X
        p = p
        # Monto Carlo Sample size
        M = M
        # traning sample size
        B = B
        # training epoches
        EPOCH = EPOCH
        # training batchsize
        BATCHSIZE = BATCHSIZE
        # \pi
        pi = torch.FloatTensor([np.pi]).cuda(0)
        # the real \eta function
        def pi_oracle(y):
            return torch.exp(1 + y) / (1 + torch.exp(1 + y))
        # working model of \eta function
        def pi_star(y):
            return torch.exp(1 - y) / (1 + torch.exp(1 - y))
        # pdf of normal distribution
        def dnorm(y):
            return torch.exp(-y ** 2 / 2) / torch.sqrt(2 * pi)
        # derivation
        def ddnorm(y):
            return torch.exp(-y ** 2 / 2) / torch.sqrt(2 * pi) * (-y)

        np.random.seed(seeds[item])
        torch.manual_seed(seeds[item])

        # dataset X,R,Y
        X = torch.normal(0.5, 0.5, (N, p))
        X[:, 0] = 1
        beta_true = torch.FloatTensor([0.25, -0.5]).reshape(2, 1)
        E = torch.normal(0, 1, (N, 1))
        Y_nm = torch.mm(X, beta_true) + E
        R = torch.FloatTensor(list(map(lambda p: np.random.binomial(1, p, 1), pi_oracle(Y_nm))))
        Y = Y_nm * R
        R = R.cuda(0)
        X = X.cuda(0)
        Y = Y.cuda(0)
        '''
        2. define neural network 
        '''

        beta = nn.Linear(p, 1, bias=False).cuda(0)

        b0 = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
            # nn.Tanh(),
            # nn.Linear(4, 2),
            # nn.Tanh(),
            # nn.Linear(2, 1)
        )
        b0 = b0.cuda(0)

        b1 = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
            # nn.Tanh(),
            # nn.Linear(4, 2),
            # nn.Tanh(),
            # nn.Linear(2, 1)
        )

        b1 = b1.cuda(0)

        # b0 = torch.load('b0_10.pkl')
        # b1 = torch.load('b1_10.pkl')
        # beta = torch.load('beta_10.pkl')

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
        beta.apply(weight_init)

        # generate training data and set mini batch
        y_train0 = torch.linspace(int(min(Y)) - 0.5, int(max(Y)) + 0.5, B).reshape(B, 1).cuda(0)
        train_data = Data.TensorDataset(y_train0, torch.zeros(B, 1).cuda(0))
        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)

        # set parameters and optimizer
        Parameters_b = [{"params": b0.parameters()}, {"params": b1.parameters()}]  # ,{"params":beta.parameters()}]
        optimizer_b = torch.optim.Adam(Parameters_b, lr=1e-3)
        Parameters_beta = beta.parameters()
        optimizer_beta = torch.optim.Adam(Parameters_beta, lr=1e-2)

        print('================================================================================')
        print('start!')
        print('================================================================================')

        ones_M = torch.ones(M, 1).cuda(0)
        ones_N = torch.ones(N, 1).cuda(0)
        ones_B = torch.ones(BATCHSIZE, 1).cuda(0)
        st = time.time()

        for epoch in range(EPOCH):
            for step, (y_train, _) in enumerate(train_loader):

                rnorm = torch.normal(0, 1, (M, 1)).cuda(0)

                t = torch.kron(ones_N, rnorm) + torch.kron(beta(X).data, ones_M)
                int_denom = 1 - torch.mean(pi_star(t.reshape(N, M)), dim=1).reshape(-1, 1) # denominator
                bb = torch.mean((torch.kron(ones_N, rnorm) * pi_star(t)).reshape(N, M), dim=1).reshape(-1, 1)
                int0 = bb * X[:, 0].reshape(-1, 1)
                int1 = bb * X[:, 1].reshape(-1, 1)
                int_b0 = torch.mean((b0(t) * pi_star(t)).reshape(N, M), dim=1).reshape(-1, 1)
                int_b1 = torch.mean((b1(t) * pi_star(t)).reshape(N, M), dim=1).reshape(-1, 1)

                kron_betaX = torch.kron(ones_B, beta(X).data)
                kron_int_denom = torch.kron(ones_B, int_denom)

                def h(y):
                    return torch.mean(dnorm(y - kron_betaX).reshape(BATCHSIZE, N), dim=1).reshape(-1, 1)

                def f0(y):
                    return torch.mean(
                        (torch.kron(ones_B, int_b0) / kron_int_denom * dnorm(y - kron_betaX)).reshape(BATCHSIZE, N),
                        dim=1).reshape(-1, 1)

                def f1(y):
                    return torch.mean(
                        (torch.kron(ones_B, int_b1) / kron_int_denom * dnorm(y - kron_betaX)).reshape(BATCHSIZE, N),
                        dim=1).reshape(-1, 1)

                def g0(y):
                    return torch.mean(
                        (ddnorm(y - kron_betaX) * torch.kron(ones_B, -X[:, 0].reshape(N, 1)) + torch.kron(ones_B,
                                                                                                          int0) / kron_int_denom * dnorm(
                            y - kron_betaX)).reshape(BATCHSIZE, N)
                        , dim=1).reshape(-1, 1)

                def g1(y):
                    return torch.mean(
                        (ddnorm(y - kron_betaX) * torch.kron(ones_B, -X[:, 1].reshape(N, 1)) + torch.kron(ones_B,
                                                                                                          int1) / kron_int_denom * dnorm(
                            y - kron_betaX)).reshape(BATCHSIZE, N)
                        , dim=1).reshape(-1, 1)

                y = torch.kron(y_train, ones_N)
                res0 = torch.mean((b0(y_train) * h(y) + f0(y) - g0(y)) ** 2)
                res1 = torch.mean((b1(y_train) * h(y) + f1(y) - g1(y)) ** 2)
                loss_b = res0 + res1

                optimizer_b.zero_grad()
                loss_b.backward()
                optimizer_b.step()

                if step % 5 == 0:
                    rnorm = torch.normal(0, 1, (M, 1)).cuda(0)

                    t = torch.kron(ones_N, rnorm) + torch.kron(beta(X), ones_M)
                    int_denom = 1 - torch.mean(pi_star(t.reshape(N, M).t()), dim=0).reshape(-1, 1)
                    bb = torch.mean((torch.kron(ones_N, rnorm) * pi_star(t)).reshape(N, M), dim=1).reshape(-1, 1)
                    int0 = bb * X[:, 0].reshape(-1, 1)
                    int1 = bb * X[:, 1].reshape(-1, 1)
                    int_b0 = torch.mean((b0(t).data * pi_star(t)).reshape(N, M), dim=1).reshape(-1, 1)
                    int_b1 = torch.mean((b1(t).data * pi_star(t)).reshape(N, M), dim=1).reshape(-1, 1)

                    S_eff0 = torch.mean(R / dnorm(Y - beta(X)) * ddnorm(Y - beta(X)) * (-X[:, 0]).reshape(-1, 1) - (
                            1 - R) / int_denom * int0 - R * b0(Y).data + (1 - R) / int_denom * int_b0)
                    S_eff1 = torch.mean(R / dnorm(Y - beta(X)) * ddnorm(Y - beta(X)) * (-X[:, 1]).reshape(-1, 1) - (
                            1 - R) / int_denom * int1 - R * b1(Y).data + (1 - R) / int_denom * int_b1)
                    loss_beta = S_eff0 ** 2 + S_eff1 ** 2

                    optimizer_beta.zero_grad()
                    loss_beta.backward()
                    optimizer_beta.step()
                if step % 25 == 0:
                    print('Epoch: ', epoch, 'Step:', step,
                          [loss_b.to('cpu').data.numpy(), loss_beta.to('cpu').data.numpy(), beta.state_dict()])

        print(beta.state_dict())
        directory = "beta_files/"

        for key, value in beta.state_dict().items():
            file_path = directory + str(key) + "mnartime.txt"
            with open(file_path, "a") as file:
                file.write(str(value) + '\n')
        et = time.time()
        print(et - st)

if __name__ == "__main__":
    # hyperparameters
    # sample size of original data
    N = 500
    # dimension of X
    p = 2
    # Monto Carlo Sample size
    M = 1000
    # traning sample size
    B = 10000
    # training epoches
    EPOCH = 20000
    # training batchsize
    BATCHSIZE = 100

    # func
    _main_(N, p, M, B, EPOCH, BATCHSIZE)