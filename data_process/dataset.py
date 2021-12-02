from PIL import Image
from torch.utils.data import DataLoader, Dataset
from config.config import *
import torch
from torchvision import transforms
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

class MyDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(MyDataset, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform

        if train:
            file = open(root + 'train.txt', 'r')
        else:
            file = open(root + 'test.txt', 'r')

        imgs = []
        for line in file:    
            line = line.rstrip()    
            imgs.append((line.split(' ')[0], int(line.split(' ')[1])))   
        self.imgs = imgs

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = Image.open(path).convert('RGB') 

        if self.transform is not None:
            img =  self.transform(img)   
        return img, label

    def __len__(self):
        return len(self.imgs)

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            #print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

def create_dataloader(is_train,dataset):
    if is_train:
        train_sampler=torch.utils.data.distributed.DistributedSampler(dataset)
        train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE,pin_memory=True,num_workers = 8, shuffle=False, drop_last=True,sampler=train_sampler)
        return train_loader
    else:
        test_sampler=torch.utils.data.distributed.DistributedSampler(dataset)
        test_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True,num_workers = 8, drop_last=False,sampler=test_sampler)
        return test_loader

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

