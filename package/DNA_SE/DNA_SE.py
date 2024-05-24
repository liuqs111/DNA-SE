import torch
from torch import nn
from torch.optim import Adam
import torch.utils.data as Data
import numpy as np
import torch.nn.functional as F    
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

def dna(p, Oi, BATCHSIZE, EPOCH, M, B, input_dim, hidden_dim, output_dim, K_func, C_func, depth, psi_func = None, esti_mask = False, activation = 'tanh', device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    '''
    p: int, the number of features X
    Oi: np.array, the observed data
    BATCHSIZE: int, the batch size
    EPOCH: int, the number of epochs
    M: int, the Monte Carlo number of data points for t
    B: int, the Monto Carlo number of data points for s
    input_dim: int, the input dimension of the model
    hidden_dim: int, the hidden dimension of the model
    output_dim: int, the output dimension of the model
    K_func: function, the kernel function
    C: function, the function to calculate the integral
    psi_func: function, the function to calculate the psi function
    depth: int, the depth of the model L
    activation: str, the activation function [relu, sigmoid, tanh]
    device: torch.device, the device to run the model
    '''
    
    # class LinearModel(nn.Module):
    #     def __init__(self, input_dim, output_dim):
    #         super(LinearModel, self).__init__()
    #         self.linear = nn.Linear(input_dim, output_dim)
    #         self.act = nn.Tanh()

    #     def forward(self, x):
    #         return self.linear(self.act(x))
        
    beta = nn.Linear(p, 1, bias=False).to(device)
    
    class MyModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, depth, activation):
            super(MyModel, self).__init__()
            layers = []
            for i in range(depth):
                if i == 0:
                    layers.append(nn.Linear(input_dim, hidden_dim))
                    if activation == 'relu':
                        layers.append(nn.ReLU())
                    elif activation == 'sigmoid':
                        layers.append(nn.Sigmoid())
                    elif activation == 'tanh':
                        layers.append(nn.Tanh())
                    else:
                        raise ValueError("Unsupported activation function.")
                elif i == depth - 1:
                    layers.append(nn.Linear(hidden_dim, output_dim))
                else:
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                     
            self.layers = nn.ModuleList(layers)
            
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model_b = MyModel(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, depth=depth, activation=activation).to(device)

    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model_b.apply(weight_init)
    beta.apply(weight_init)
    
    # data loader
    o_train = torch.hstack([torch.linspace(int(min(Oi)) - 0.5, int(max(Oi)) + 0.5, B).reshape(B, 1) for _ in range(p + 1)]).to(device) # o_train is a tensor of shape (B, p+1) and the first p-th is for Oi, the last is for t
    train_data = Data.TensorDataset(o_train, torch.zeros(B, 1).to(device))
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)
    
    # set parameters and optimizer
    Parameters_b = model_b.parameters() # ,{"params":beta.parameters()}]
    optimizer_b = torch.optim.Adam(Parameters_b, lr=1e-3)
    Parameters_beta = beta.parameters()
    optimizer_beta = torch.optim.Adam(Parameters_beta, lr=1e-2)
    losses = []
    epochs = []
    
    for epoch in range(EPOCH):
        for step, (t_train, _) in enumerate(train_loader):
            
            t = torch.linspace(int(min(Oi)),int(max(Oi)),M).reshape(M,1).to(device)
            s = t_train[..., -1].reshape(BATCHSIZE,1).to(device)
            ones_M = torch.ones(M,1).to(device)

            def loss_function_integral(s, t, Oi, beta, model_b):
                
                inter_left = torch.mean(K_func(s,t,Oi,beta)*torch.kron(ones_M,model_b(s)).reshape(M,BATCHSIZE), dim=1).reshape(M,1)
                inter_right = C_func(t,Oi,beta) + model_b(t)
                return torch.mean((inter_left - inter_right)**2)
            
            optimizer_b.zero_grad()
            loss_integral = loss_function_integral(s, t, Oi, beta, model_b)
            loss_integral.backward()
            optimizer_b.step()

            
            if step % 5 == 0 and esti_mask == 1:
                
                def loss_function_estimate(Oi, beta, model_b):
                
                    return torch.mean(psi_func(Oi, beta, model_b)**2)
            
                optimizer_beta.zero_grad()
                loss_estimate = loss_function_estimate(Oi, beta, model_b)
                loss_estimate.backward()
                optimizer_beta.step()

        if epoch % 100 == 0 and esti_mask == 1:
            print(f'Epoch {epoch}: loss = {loss_estimate.item()}')
        if epoch % 1 == 0 and esti_mask == 0:
            print(f'Epoch {epoch}: loss = {loss_integral.item()}')
            losses.append(loss_integral.item())
            epochs.append(epoch)

    if esti_mask == 1:
        return model_b, beta.data()
    else:
        return model_b

# a simple example
# p = 1
# Oi = torch.rand(1000,1)
# BATCHSIZE = 100
# EPOCH = 200
# M = 1000
# B = 10000
# input_dim = 1
# output_dim = 1
# hidden_dim = 5
# depth = 2
# ones_B = torch.ones(M,1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# def K_func(x,y,Oi,b):
#     return y * torch.kron(ones_B,x).reshape(BATCHSIZE,M)
# def C_func(x,Oi,b):
#     return -torch.exp(x)
# output = dna(p, Oi, BATCHSIZE, EPOCH, M, B, input_dim, hidden_dim, output_dim, K_func, C_func, depth)
# print(output)

'''
beta.data() is used to get the value of beta
'''

# Path: DNA-SE/package/DNA_SE/DNA_SE.py
