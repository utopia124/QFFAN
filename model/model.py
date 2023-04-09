import math
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
        self.embedding = Embedding(concepts_dict)
        self.mlp = MLP()
        self.concepts = concepts_dict
        self.transformer_encoder = Transformer()

    def forward(self, x):
        d_k = 1024
        shot_num_split = data_set_config.shot_num_split
        frames_per_shot = data_set_config.frames_per_shot
        google_net_out_channel = data_set_config.google_net_out_channel
        embedding_size = self.embedding.embedding_size

        frame_fea = x[0].to(device)
        batch_size = frame_fea.shape[0]
        frame_fea = torch.permute(frame_fea, [0, 2, 1, 3])
        frame_fea = frame_fea.view(batch_size, shot_num_split, frames_per_shot, google_net_out_channel)
        frame_fea_with_attention = []
        for i in range(shot_num_split):
            fea = frame_fea[:, i, :, :]
            fea = self.transformer_encoder(fea)
            frame_fea_with_attention.append(fea)
        frame_fea_with_attention = torch.stack(frame_fea_with_attention, dim=1).view(batch_size* shot_num_split* frames_per_shot,google_net_out_channel)
        concept_pair = x[1]
        query_emb = self.embedding.forward(concept_pair)

        weight_q = torch.nn.Parameter(torch.randn(embedding_size, d_k), requires_grad=True).to(device)
        weight_k = torch.nn.Parameter(torch.randn(google_net_out_channel, d_k), requires_grad=True).to(device)
        weight_v = torch.nn.Parameter(torch.randn(google_net_out_channel, d_k), requires_grad=True).to(device)

        query_emb = query_emb.view(2 * batch_size, embedding_size)
        Q = torch.matmul(query_emb, weight_q)
        K = torch.matmul(frame_fea_with_attention, weight_k)
        V = torch.matmul(frame_fea_with_attention, weight_v)

        Q = Q.view(batch_size, 2, d_k)
        K = K.view(batch_size, shot_num_split * frames_per_shot, d_k)
        V = V.view(batch_size, shot_num_split, frames_per_shot, d_k)

        attention = []
        softmax = nn.Softmax(dim=2)
        for i in range(batch_size):
            tmp1 = torch.matmul(Q[i], K[i].transpose(0, 1)) / math.sqrt(d_k)
            tmp1 = tmp1.view(2, shot_num_split, frames_per_shot)
            tmp1 = softmax.forward(tmp1)
            score = tmp1.view(2, shot_num_split, frames_per_shot)
            attention_batch = []
            for j in range(score.shape[1]):
                c = torch.matmul(score[:, j, :], V[i, j, :, :]).flatten()
                attention_batch.append(c)
            attention_batch = torch.stack(attention_batch, dim=0)
            attention.append(attention_batch)
        attention = torch.stack(attention, dim=0)
        out = self.mlp.forward(attention, batch_size=batch_size)
        return out


class Embedding(nn.Module):
    def __init__(self, concepts_dict):
        super().__init__()
        self.embedding_size = 256
        self.concepts = concepts_dict
        self.model = nn.Embedding(len(self.concepts), self.embedding_size)

    def forward(self, x):
        con1, con2 = x
        batch_size = len(con1)
        emb_batch = []
        for i in range(batch_size):
            emb1 = self.model(torch.tensor(self.concepts[con1[i]], device=device))
            emb2 = self.model(torch.tensor(self.concepts[con2[i]], device=device))
            emb = torch.stack((emb1, emb2), dim=0)
            emb = torch.unsqueeze(emb, dim=0)
            emb_batch.append(emb)
        emb_batch = torch.stack(emb_batch, dim=0)
        return emb_batch


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=8, batch_first=True)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, x):
        out = self.model(x)
        return out


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048, device=device),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024, device=device),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1, device=device),
            nn.Sigmoid()
        )

    def forward(self, x, batch_size):
        shot_num = x.shape[1]
        out = self.model(x).squeeze()

        return out
