import torch
import torch.nn as nn
from .resnet import resnet50
import torchvision

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.backbone = resnet50()
    def forward(self, x):
        x = self.backbone(x)
        return x



def ResNet50(pretrained=True,num_classes=200):
    model = torchvision.models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(2048,num_classes)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias,0)
    return model



if __name__ == "__main__":

    model = Classifier()

    model.cuda()
    print(model)
    data = torch.ones((1, 3, 224, 224)).cuda()
    output = model(data)
    print(output.shape)

