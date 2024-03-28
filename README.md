# DNA-SE: Towards Deep Neural-Net Assisted Semiparametric Estimation
DNA-SE is an approach for solving the parameter of interest in semi-parametric. We give 3 examples about missing not at random, sensitivity analysis in causal inference and transfer learning. DNA-SE proposes a method using deep neural network to estimate or calculate the parameters with the solution given by integral equation. Also it has a iterative alternating procedure with Monte Carlo integration and a new loss function.

## Setup
For the requirments, the DNA-SE methods depend on python>=3.7, torch>=1.12, time package.

Using the following command in Python to install:
```
conda create -n --envname python>=3.7
conda activate --envname
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Usage
The command of function for MNAR is:
```
from mnar import _main_
_main_(N,p,J,B,EPOCH,BATCHSIZE)
```
For the parameters, we have:

```N```: the sample of original data;

```p```: the dimension of $X$;

```J```: Monte Carlo Sample Size;

```B```: training sample size;

```EPOCH```: training epoch;

```BATCHSIZE```: training batchsize. 

## Values
```print('Epoch: ', epoch, 'Step:', step,[loss_omega.to('cpu').data.numpy(), loss_beta.to('cpu').data.numpy(), beta.state_dict()])```

The output of function is ```Epoch```, ```step```, the loss function of $\mathbf{\omega}$ and $\mathbf{\beta}$ and the interested parameter $\mathbf{\beta}$.

## Details
The details of figures and paper are in the fold and you can check it.
