from model.model import QueryFocusedFullyAttentionNetwork
from data.ute_video_dataset import UTEVideoDataset
import torch
import torch.nn as nn
import torch.nn.functional as F

dataset = UTEVideoDataset("D://Workspace//Data//UTE_video")

net = QueryFocusedFullyAttentionNetwork(dataset.concepts)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_, ground_truth = dataset[0]
frames, concepts = in_
# print(net.concepts)
# net.test()
with torch.no_grad():
    pred = net.forward(dataset[0])
    loss = F.cross_entropy(pred, torch.tensor(ground_truth,device=device))
    print(loss)
# print(dataset[0][0])
