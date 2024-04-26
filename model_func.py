import torch
from torch import nn
from torch.optim import Adam
import torch.utils.data as Data
import numpy as np
import torch.nn.functional as F     

def train_model_b(p, Oi, BATCHSIZE, EPOCH, M, B, device, input_dim, output_dim, K_func, C, psi_func):
    
    class LinearModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LinearModel, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.linear(x)
        
    beta = nn.Linear(p, 1, bias=False).cuda(0)

    model_b = LinearModel(input_dim, output_dim)

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
    o_train = torch.linspace(int(min(Oi)) - 0.5, int(max(Oi)) + 0.5, B).reshape(B, 1).to(device)
    train_data = Data.TensorDataset(o_train, torch.zeros(B, 1).to(device))
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)
    
    # set parameters and optimizer
    Parameters_b = model_b.parameters() # ,{"params":beta.parameters()}]
    optimizer_b = torch.optim.Adam(Parameters_b, lr=1e-3)
    Parameters_beta = beta.parameters()
    optimizer_beta = torch.optim.Adam(Parameters_beta, lr=1e-2)
    
    for epoch in range(EPOCH):
        for step, (t_train, _) in enumerate(train_loader):
            
            s = torch.linspace(min(Oi),max(Oi),M).reshape(M,1).cuda(0)
            t = t_train

            def loss_function_integral(s, t, Oi, beta, model_b):
                
                inter_left = torch.sum(K_func(s,t,Oi,beta)*model_b(s,Oi))
                inter_right = C(t,Oi,beta)+model_b(t,Oi)
                return torch.mean(torch.sum((inter_left - inter_right)**2))
            
            optimizer_b.zero_grad()
            loss_integral = loss_function_integral(s, t_train, Oi, beta, model_b)
            loss_integral.backward()
            optimizer_beta.step()
            
            if step % 5 == 0:
                
                def loss_function_estimate(Oi, beta, model_b):
                
                    return torch.mean(psi_func(Oi, beta, model_b))**2
            
                optimizer_beta.zero_grad()
                loss_estimate = loss_function_estimate(t_train, Oi, beta, model_b)
                loss_estimate.backward()
                optimizer_beta.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}: loss = {loss_estimate.item()}')
    return model_b, beta.data(), optimizer_b, optimizer_beta