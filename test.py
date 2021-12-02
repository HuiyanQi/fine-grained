import torch
from config.config import *
from operation.operation import can
from data_process.dataset import accuracy
from config.config import parse


def test(model,test_loader):
    args = parse()
    total_test_acc = 0
    with torch.no_grad():
        for i,test_data in enumerate(test_loader):
            imgs, targets = test_data
            imgs, targets = imgs.cuda(), targets.cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            acc = accuracy(outputs, targets)[0]
            #acc = can(outputs=outputs,targets=targets,prior=train_prior)
            total_test_acc += acc
            if args.local_rank == 0:
                if i % 10 == 0:
                    print("         [{}/{}], Loss:{} acc:{}".format(i, len(test_loader), format(loss.item(), '.5f'), acc))
    return total_test_acc
