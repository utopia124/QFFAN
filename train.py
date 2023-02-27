from model.model import QueryFocusedFullyAttentionNetwork
from data.ute_video_dataset import UTEVideoDataset

dataset = UTEVideoDataset("D://Workspace//Data//UTE_video")

net = QueryFocusedFullyAttentionNetwork()

net.forward(dataset[0])

# print(dataset[0][0])
