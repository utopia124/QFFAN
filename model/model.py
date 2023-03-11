import torch
import torch.nn as nn
from torchvision import models
import cv2
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.io import ImageReadMode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QueryFocusedFullyAttentionNetwork(nn.Module):
    def __init__(self, concepts_dict):
        super().__init__()
        self.resnet = ResNet34().to(device)
        self.transformer_frame = Transformer1().to(device)
        self.transformer_shot = Transformer2().to(device)
        self.embedding = Embedding(concepts_dict).to(device)
        self.fusion = Fusion().to(device)
        self.mlp = MLP().to(device)
        self.concepts = concepts_dict

    # def test(self):
    #     print(self.concepts['Shoes'])
    #

    def forward(self, x):
        in_, pred = x
        frames, concepts = in_
        # trans = transforms.ToTensor()
        img_fea = []
        a = 1
        for raw_img in frames:
            # ndarray_img = cv2.imread(raw_img)
            img = read_image(raw_img, mode=ImageReadMode.RGB).to(device)
            img = img.float()
            # print(img.dtype)
            # img = torch.FloatTensor(img)
            # print(img.shape)
            # tensor_img1 = trans(ndarray_img)
            # print("tensor_img1:{}".format(tensor_img1.shape))
            # tensor_img2 = torch.tensor(ndarray_img).transpose(0, 1)
            # tensor_img2 = tensor_img2.transpose(0, 2).to(device)
            # print("tensor_img2:{}".format(tensor_img2.shape))
            img_tensor = torch.unsqueeze(img, dim=0)
            # print(img_tensor.device)
            # img_tensor2 = torch.unsqueeze(tensor_img2, dim=0)
            # print("img_tensor:{}".format(img_tensor.shape))
            # print("img_tensor2:{}".format(img_tensor2.shape))
            resnet_out = self.resnet.forward(img_tensor)
            # print(img_tensor.device)
            # print(resnet_out.shape)
            print("\r{}".format(a), end="", flush=True)
            a = a + 1
            img_fea.append(resnet_out)
        res = 5 - len(img_fea) % 5
        for i in range(res):
            img_fea.append(torch.zeros(1, 200, dtype=torch.float, device=device))
        img_fea = torch.stack(img_fea, dim=0)
        # print(img_fea.shape)

        shot_features_len = len(img_fea) // 5
        # print(shot_features_len)
        shot_features = []
        for i in range(shot_features_len):
            right = 5 * i
            left = 5 * (i + 1)
            out = self.transformer_frame(img_fea[right:left])
            out = torch.flatten(out)
            # print(out.shape)
            shot_features.append(out)
        # out = self.transformer_frame(img_fea[0:5])
        shot_features = torch.stack(shot_features, dim=0)
        shot_features = torch.unsqueeze(shot_features, dim=1)
        out = self.transformer_shot.forward(shot_features)
        # print(out)
        # print(out.shape)

        emb = self.embedding.forward(concepts).flatten().to(device)
        shot_select = []
        for eu in out:
            # print(eu.shape)
            # print(emb.shape)
            # mlp_in = torch.cat((eu.flatten(), emb), dim=0)
            mlp_in = self.fusion.forward(eu.flatten(), emb)
            # print(mlp_in.shape)
            out = self.mlp.forward(mlp_in)
            shot_select.append(out[0])
        shot_select = torch.stack(shot_select, dim=0)
        shot_select = shot_select.flatten()
        # print(shot_select.shape)
        # print(len(pred))
        return shot_select


class Embedding(nn.Module):
    def __init__(self, concepts_dict):
        super().__init__()
        self.concepts = concepts_dict
        self.model = nn.Embedding(48, 200)

    def forward(self, x):
        con1, con2 = x
        emb = self.model(torch.tensor([self.concepts[con1], self.concepts[con2]], device=device))
        # print(emb.shape)
        return emb


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet34()
        self.model.fc = nn.Linear(self.model.fc.in_features, 200)

    def forward(self, x):
        out = self.model(x)
        return out


class Transformer1(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=200, nhead=8)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, x):
        out = self.model(x)
        return out


class Transformer2(nn.Module):
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
        self.hello = "hello"

    def forward(self, x, z):
        self.hello = "world"
        out = torch.cat((x, z), dim=0)
        return out


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1400, 512, device=device),
            nn.ReLU(),
            nn.Linear(512, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, 32, device=device),
            nn.ReLU(),
            nn.Linear(32, 2, device=device),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        out = self.model(x)
        return out
