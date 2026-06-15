import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import LRScheduler

__all__ = ['WarmUpCosineAnnealingLR', 'FakeLR', 'L1ALoss']


class WarmUpCosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        else:
            k = 1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]


class FakeLR(LRScheduler):
    def __init__(self, optimizer):
        super(FakeLR, self).__init__(optimizer=optimizer)

    def get_lr(self):
        return self.base_lrs
    

class L1ALoss(nn.Module):
    def __init__(self, mode='const', lamda_start=1e-2, lamda_end=0, lamda_max=5e-2, T_max=1000, alpha=0):
        super().__init__()
        self.mode = mode
        self.lamda_start = lamda_start
        self.lamda_max = lamda_max
        self.lamda_end = lamda_end
        self.T_max = T_max
        self.lamda = self.lamda_start
        self.alpha = alpha

    def forward(self, pred, label, codeword, codewordQ):
        mse1 = nn.MSELoss()
        mse2 = nn.MSELoss()
        norm = torch.norm(codeword - 0.5,p=1)/(codeword.numel()*0.5)
        loss = mse1(label, pred) + self.lamda*mse2(codeword, codewordQ) + self.alpha*norm
        return loss
