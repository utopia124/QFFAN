import torch


class ModelConfig:
    def __init__(self):
        self.res_net_size = 200


class TrainingConfig:
    def __init__(self):
        self.device = torch.device("cuda")


class DatasetConfig:
    def __init__(self):
        self.split = 0.2
        self.shot_num_split = 10
        self.frames_per_shot = 5
        self.google_net_out_channel = 1000
