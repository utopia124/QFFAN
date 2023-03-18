from model.model import QueryFocusedFullyAttentionNetwork
from data.ute_video_dataset import UTEVideoDataset
import torch.nn as nn
import torch.optim as optim
from config.config import TrainingConfig

training_config = TrainingConfig()
dataset = UTEVideoDataset("D://Workspace//Data//UTE_video")
net = QueryFocusedFullyAttentionNetwork(dataset.concepts)


def train():
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(net.parameters(), lr=learning_rate)
    split = 0.2


def forward_and_loss():
    in_, ground_truth = dataset[0]
    frames, concepts = in_
    # print(net.concepts)
    # net.test()
    with torch.no_grad():
        pred = net.forward(dataset[0])
        loss = F.cross_entropy(pred, torch.tensor(ground_truth, device=training_config.device))
        print(loss)
