import time
from config import config
from train import train
from test import test
from model.model import ResNet50
from data_process.dataset import MyDataset,create_dataloader,Cub2011
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from config.config import parse
from operation.operation import EMA
#import torch.cuda.amp.autocast as aotucast

#import torch.cuda.amp.GradScaler as GradScaler

#if not os.path.exists(datadir_path):
    #data_split()
# import pdb
# pdb.set_trace()

args = parse()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group('nccl',init_method='env://')

#train_dataset = MyDataset(root = config.datadir_path,train=True,transform=config.train_transform)
#test_dataset = MyDataset(root = config.datadir_path,train=False,transform=config.test_transform)



#train_prior = prior().cuda()


train_dataset = Cub2011(root=config.root, train=True, transform=config.train_transform)
test_dataset = Cub2011(root=config.root, train=False, transform=config.test_transform)
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
train_loader = create_dataloader(True, train_dataset)
test_loader = create_dataloader(False, test_dataset)


total_iters = len(train_loader) * config.epoch


#model = Classifier()
model = ResNet50()
#model = convert_syncbn_model(model)
device=torch.device('cuda:{}'.format(args.local_rank))
model=model.to(device)
ema = EMA(model, 0.999)
ema.register()
#if torch.cuda.is_available():
#    model.cuda()
#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")
#    model = nn.DataParallel(model)


optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay,momentum=config.momentum)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
loss_fn = nn.CrossEntropyLoss()
best_acc = -1
model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
#scaler = GradScaler()



for i in range(config.epoch):
    if args.local_rank == 0:
        print("Epoch:[{}]".format(i + 1))
    model.train()
    if args.local_rank == 0:
        print("    train:")
    time_start = time.time()
    total_train_loss = train(model, optimizer, train_loader, scheduler, ema)
    time_end = time.time()
    if args.local_rank == 0:
        print("         train loss:{},The elapsed time:{}s".format(
            format(total_train_loss / len(train_loader), '.5f'),
            time_end - time_start))
    ema.apply_shadow()
    model.eval()
    ema.restore()
    if args.local_rank == 0:
        print("    test:")
    time_start = time.time()
    total_test_acc = test(model, test_loader)
    time_end = time.time()
    if total_test_acc / len(test_loader) > best_acc:
        best_acc = total_test_acc / len(test_loader)
    if args.local_rank == 7:
        print("Epoch:[{}]   test acc = {}, best acc = {}, The elapsed time:{}s.\n".format(i + 1,format(total_test_acc.item() / len(test_loader),'.2f'),best_acc.item(),time_end - time_start))
    # scheduler.step()


