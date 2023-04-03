import torch
import torch.nn as nn
from config.config import DatasetConfig
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_set_config = DatasetConfig()


class QueryFocusedFullyAttentionNetwork(nn.Module):
    def __init__(self, concepts_dict):
        super().__init__()
        self.pool = nn.MaxPool1d(5, 5)
        self.transformer_frame = Transformer1()
        self.transformer_shot = Transformer2()
        self.embedding = Embedding(concepts_dict)
        self.fusion = Fusion()
        self.mlp = MLP()
        self.concepts = concepts_dict

    def forward(self, x):
        frames_fea = torch.squeeze(x[0], dim=2)
        concept_pair = x[1]
        shot_features = []
        # print(frames_fea.shape)
        fea_list = torch.split(frames_fea, data_set_config.frames_per_shot, dim=1)
        for fea in fea_list:
            # print(fea.shape)
            fea = fea.to('cuda')
            # print(fea.shape)
            out = self.transformer_frame(fea)
            # print("transformer_frame:{}".format(out.shape))
            shot_features.append(out)
        # for i in range(data_set_config.shot_num_split):
        #     right = data_set_config.frames_per_shot * i
        #     left = data_set_config.frames_per_shot * (i + 1)
        #     out = self.transformer_frame(frames_fea[right:left])
        #     # print(out.shape)
        #     out = torch.flatten(out)
        #     shot_features.append(out)
        shot_features = torch.stack(shot_features, dim=0)
        shot_features = torch.flatten(shot_features, start_dim=2, end_dim=3)

        shot_features = self.pool(shot_features)
        # print(shot_features.shape)
        # shot_features = torch.unsqueeze(shot_features, dim=1)
        batch_size = shot_features.shape[1]
        # print("shot_features.shape:{}".format(shot_features.shape))
        emb = self.embedding.forward(concept_pair).flatten().to(device)
        emb = emb.repeat(data_set_config.shot_num_split, batch_size, 1)
        # print(emb.shape)
        shot_features = torch.cat([shot_features, emb], dim=2)
        # print(shot_features.shape)
        out = self.transformer_shot.forward(shot_features)
        # print(out.shape)

        shot_select = []
        for eu in out:
            # mlp_in = self.fusion.forward(eu.flatten(), emb)
            out = self.mlp.forward(eu)
            shot_select.append(out[:, 0])
        shot_select = torch.stack(shot_select, dim=1)
        # print("shot_select.shape{}".format(shot_select.shape))
        # shot_select = shot_select.flatten()
        # print(shot_select.shape)
        return shot_select


class Embedding(nn.Module):
    def __init__(self, concepts_dict):
        super().__init__()
        self.concepts = concepts_dict
        self.model = nn.Embedding(48, 200)

    def forward(self, x):
        con1, con2 = x
        emb = self.model(torch.tensor([self.concepts[con1[0]], self.concepts[con2[0]]], device=device))
        # print(emb.shape)
        return emb


class ConvShot(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel = data_set_config.shot_num_split
        self.model = nn.Sequential(
            nn.Conv1d(self.channel, self.channel, )
        )


class ConvSelect(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel = data_set_config.shot_num_split
        self.model = nn.Sequential(
            nn.Conv1d(self.channel, self.channel, kernel_size=5),
            nn.MaxPool1d(2),
        )


class Transformer1(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=8, batch_first=True)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, x):
        out = self.model(x)
        return out


class Transformer2(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=1400, nhead=8)
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
            nn.Softmax()
        )

    def forward(self, x):
        out = self.model(x)
        return out
