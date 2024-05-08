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

For the usage of this package, you should first download the github repositories into your server. The command of this is:

```
git clone https://github.com/liuqs111/DNA-SE.git
```

Then enter the path of model in this file

```
cd model
```

And users can use the function by running the command below in command line:

```
from mymodel import model_b_training
```
