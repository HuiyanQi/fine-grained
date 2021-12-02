from torch import nn
from config import config
from config import config
import torch
from data_process.dataset import accuracy
def smooth(targets: torch.Tensor, classes: int, smoothing):
    targets = targets.resize_(config.BATCH_SIZE, 1)
    one_hot = torch.zeros(config.BATCH_SIZE, config.class_num).scatter_(1, targets, 1)

    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((one_hot.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape)    
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(one_hot, 1)
        true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)
    return true_dist


def celoss(outputs,targets):
    logsoftmax_func = nn.LogSoftmax(dim=1)
    outputs = logsoftmax_func(outputs)
    loss = targets.mul(outputs)
    loss = -torch.sum(loss, dim = 1).mean()
    return loss


def prior():
    images_class_labels = []
    with open(config.path_image_class_labels, 'r') as f1:
        for line in f1:
            images_class_labels.append(list(line.strip('\n').split(',')))

    split = []
    with open(config.path_split, 'r') as f2:
        for line in f2:
            split.append(list(line.strip('\n').split(',')))

    num = len(images_class_labels)
    prior = torch.zeros(config.class_num)
    for k in range(num):
        if int(split[k][0][-1]) == 1:
            prior[int(images_class_labels[k][0].split(' ')[1])-1]+=1.

    prior /= prior.sum()

    return prior

def can(outputs,targets,prior):
    k = torch.tensor(config.top_k)
    softmax_func = nn.Softmax(dim=1)
    outputs = softmax_func(outputs)
    y_pred_topk = torch.sort(outputs, axis=1).values[:, -k:]
    y_pred_topk /= y_pred_topk.sum(axis=1, keepdims=True)
    y_pred_uncertainty = -(y_pred_topk *torch.log(y_pred_topk)).sum(1) / torch.log(k)
    threshold = config.threshold
    y_pred_confident = outputs[y_pred_uncertainty < threshold]
    y_pred_unconfident = outputs[y_pred_uncertainty >= threshold]
    y_true_confident = targets[y_pred_uncertainty < threshold]
    y_true_unconfident = targets[y_pred_uncertainty >= threshold]
    
    if len(y_true_confident)==0:
        num_confident = 0.
    else:    
        num_confident = accuracy(y_pred_confident,y_true_confident)[0].mul_(len(y_true_confident) / 100.0)
    


    right, alpha, iters = 0, 1, 1
    for i, y in enumerate(y_pred_unconfident):
        Y = torch.cat([y_pred_confident, y[None]], axis=0)
        for j in range(iters):
            Y = Y ** alpha
            Y /= Y.sum(axis=0, keepdims=True)
            Y *= prior[None]
            Y /= Y.sum(axis=1, keepdims=True)
        y = Y[-1]
        if y.argmax() == y_true_unconfident[i]:
            right += 1
    acc_final = (num_confident + right)*100 / len(targets)
    return acc_final


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


