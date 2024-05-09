# DNA-SE: Towards Deep Neural-Net Assisted Semiparametric Estimation

DNA-SE is an approach for solving the parameter of interest in semi-parametric. We give 3 examples about missing not at random, sensitivity analysis in causal inference and transfer learning. DNA-SE proposes a method using deep neural network to estimate or calculate the parameters with the solution given by integral equation. Also it has a iterative alternating procedure with Monte Carlo integration and a new loss function. Furthermore, we support a python package with pytorch to use our algorithm directly.

## Setup

For the requirments, the DNA-SE methods depend on python>=3.7, torch>=1.12, time package.

Using the following command in Python to install:

```
conda create -n --envname python>=3.7
conda activate --envname
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Figures and Networks

For our method, we choose simple neural networks and prove it is useful to solve integral equations. And we suppose the bi-level algorithm which is shown in 

## Usage

The specific three examples for MNAR, Sensitivity analysis and Transfer learning, we give the codes in mnar.py, sensitivity_simu.py and transfer_learning.py which are available for you to reproduce our results.

Also in order to use our algorithm more easily, we give a simple package in python and you can check the file model.

For the usage of this package, you should first install the package *DNA_SE* into your server. The command of this is:

```
pip install DNA_SE
```
Then you can use the function *dnase* in this package:
```
from DNA_SE import dnase
```
## Alternating Usage
Also if you want to modify the code by yourself, you can choose to download the repository from this github:
```
git clone https://github.com/liuqs111/DNA-SE.git
```
Then enter the path of model in this file:
```
cd DNA_SE/package/DNA_SE
```
And you can use the function by running the command below in command line:
```
from DNA_SE import dnase
```
## Function Usage
```
dnase(p, Oi, BATCHSIZE, EPOCH, M, B, input_dim, hidden_dim, output_dim, K_func, C_func, psi_func, depth, activation = 'tanh', device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
```
`p`: int, the number of features X;

`Oi`: np.array, the observed data;

`BATCHSIZE`: int, the batch size;

`EPOCH`: int, the number of epochs;

`M`: int, the Monte Carlo number of data points for t;

`B`: int, the Monto Carlo number of data points for s;

`input_dim`: int, the input dimension of the model;

`hidden_dim`: int, the hidden dimension of the model;

`output_dim`: int, the output dimension of the model;

`K_func`: function, the kernel function;

`C_func`: function, the function to calculate the integral;

`psi_func`: function, the function to calculate the psi function;

`depth`: int, the depth of the model L;

`activation`: str, the activation function [relu, sigmoid, tanh];

`device`: torch.device, the device to run the model.
## Value
```
return model_b, beta.data(), optimizer_b, optimizer_beta
```
where beta.data() is used to get the value of beta. Also you can check the model and optimizer and choose as you like.
## Details
For the choice of parameters, for our three examples we choose tanh() as our activate function and the dimension of input, hidden and output are chosen by grid search. Also the original data O depends on yourself. You can modify the dataload of training data to satisfy different requirements, for example, using all original data or part original data given in datasets. Therefore we recommand you to download the code from this github and modify as you like. Also you can choose more deep networks to train your model which may perform better for your specific calculation.

For our three examples, since their functions are so complex, therefore we give the three unique python files for you to check and reproduce.
## Reference:
Qinshuo Liu, Zixin Wang, Xi-An Li, Xinyao Ji, Lei Zhang, LL#, and Zhonghua Liu#. DNA-SE: Towards Deep Neural-Nets Assisted Semiparametric Estimation. (2024). International Conference on Machine Learning.
## Acknowledgement
The authors would like to thank BaoLuo Sun for helpful dis-
cussions. Lin Liu is supported by NSFC Grant No.12101397
and No.12090024, Shanghai Science and Technology Com-
mission Grant No.21ZR1431000 and No.21JC1402900, and
Shanghai Municipal Science and Technology Major Project
No.2021SHZDZX0102. Zixin Wang and Lei Zhang are also
supported by Shanghai Science and Technology Commis-
sion Grant No.21ZR1431000. Lin Liu is also affiliated with
Shanghai Artificial Intelligence Laboratory and the Smart
Justice Lab of Koguan Law School of SJTU. Zhonghua Liu
is also affiliated with Columbia University Data Science
Institute.
