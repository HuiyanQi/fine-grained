import torch
from config.config import *


def judge_lr(i, lr):
    i = int(i / 30)
    if i == 0:
        return lr
    else:
        lr = lr * (0.1 ** i)
        return lr


def create_optimizer(optim,lr,model):
    if optim == 'sgd':
        #for name, param in model.named_parameters():
            #if 'fc' not in name:
                #param.requires_grad = False
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,momentum=momentum)
    else:
        raise ValueError()
    return optimizer
