from torch import nn
from torchvision import transforms
import torchvision
import argparse



ROOT_PATH = '/workspace/qhy/CUB/CUB_200_2011/'
#ROOT_PATH = '/CUB/CUB_200_2011/'
#ROOT_PATH = 'D:/qhy/CUB/CUB_200_2011/'
TRAIN_PATH = ROOT_PATH + 'dataset/train/'
TEST_PATH = ROOT_PATH + 'dataset/test/'

datadir_path = ROOT_PATH + 'dataset/'
path_images = ROOT_PATH + 'images.txt'
path_split = ROOT_PATH + 'train_test_split.txt'
path_image_class_labels = ROOT_PATH + 'image_class_labels.txt'
path_classes = ROOT_PATH + 'classes.txt'
trian_save_path = ROOT_PATH + 'dataset/train/'
test_save_path = ROOT_PATH + 'dataset/test/'
train_path = ROOT_PATH +'dataset/train.txt'
test_path = ROOT_PATH +'dataset/test.txt'

root = '/workspace/qhy/CUB/'


def parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--local_rank',type=int, default=0) 
    args = parser.parse_args()
    return args

class_num = 200
loss_fn = nn.CrossEntropyLoss()
BATCH_SIZE = 64
lr =0.1*BATCH_SIZE/256
momentum = 0.9
weight_decay = 1e-4
train_optimizer = 'sgd'
epoch = 200
top_k = 3
threshold = 0.98

train_transform = torchvision.transforms.Compose([
    transforms.Resize((512,512)),
    transforms.RandomCrop(448),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


test_transform =transforms.Compose([transforms.Resize((512,512)),
                                   transforms.CenterCrop(448),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])