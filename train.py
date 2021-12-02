from data_process.dataset import accuracy
from config import config
from operation.operation import smooth,celoss
#from torch.cuda.amp import autocast as autocast
from config.config import parse

def train(model,optimizer,train_loader,scheduler,ema):
    args = parse()
    total_train_acc = 0
    total_train_loss = 0
    for i, train_data in enumerate(train_loader):
        imgs, targets = train_data
        #imgs, targets = imgs.cuda(),targets.cuda()
        target = smooth(targets,config.class_num,smoothing = 0.1)
        imgs, target,targets = imgs.cuda(), target.cuda(),targets.cuda()
        optimizer.zero_grad()
        #with autocast():
        outputs = model(imgs)
        #loss = config.loss_fn(outputs, targets)
        loss = celoss(outputs,target)
        loss.backward()
        optimizer.step()
        ema.update()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        scheduler.step()
        acc = accuracy(outputs, targets)[0]
        total_train_acc += acc
        total_train_loss += loss.item()
        if args.local_rank == 0:
            if i % 10 == 0:
                print("         [{}/{}] lr:{} Loss:{} acc:{}".format(i, len(train_loader),
                                                                 optimizer.param_groups[-1]['lr'],
                                                                 format(loss.item(), '.5f'), acc))

    return total_train_loss

