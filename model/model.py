import torch
import torch.nn as nn
from torchvision import models
import cv2
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QueryFocusedFullyAttentionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ResNet34().cuda(device)
        self.layer2 = Transformer().cuda(device)
        self.layer3 = Transformer()
        self.layer4 = Fusion()
        self.layer5 = MLP()

    def forward(self, x):
        in_, pred = x
        frames, concepts = in_
        trans = transforms.ToTensor()
        img_fea = []
        for img in frames:
            img = cv2.imread(img)
            img = trans(img).cuda(device)
            img_tensor = torch.unsqueeze(img, dim=0)
            print(img_tensor.device)
            resnet_out = self.layer1.forward(img_tensor)
            # print(resnet_out.shape)
            img_fea.append(resnet_out)
        img_fea = torch.stack(img_fea, dim=0)
        print(img_fea.shape)
        out = self.layer2(img_fea[0:5])
        # print(out)
        return out


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet34()

    def forward(self, x):
        out = self.model(x)
        return out


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=8)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, x):
        out = self.model(x)
        return out


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
