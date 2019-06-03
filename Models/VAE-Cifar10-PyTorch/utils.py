import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="2,3" 
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F


def BCE_KLD_Loss(recon_x, x, mu, logvar) -> Variable:
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3*32*32), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD