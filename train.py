import os.path

from model.model import QueryFocusedFullyAttentionNetwork
from data.ute_video_dataset import UTEVideoDataset
from config.config import TrainingConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import math


def rule(epoch):
    lamda = math.pow(0.5, int(epoch / 5))
    return lamda


class SparseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, y, lamda):
        batch_size = len(pred)
        loss = torch.zeros(1).to('cuda')
        for i in range(batch_size):
            tmp_loss = torch.zeros(1).to('cuda')
            n = len(pred[i])
            for j, yj in enumerate(y[i]):
                if yj == 0:
                    tmp_loss += -torch.log(1 - pred[i][j] + 1e-10)
                else:
                    tmp_loss += -lamda * torch.log(pred[i][j] + 1e-10)
            tmp_loss = tmp_loss / n
            loss += tmp_loss
        loss = loss / batch_size
        return loss


training_config = TrainingConfig()

dataset = UTEVideoDataset("D://Workspace//Data//UTE_video")
# 划分数据集
split = 0.8
train_size = int(split * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# 获取dataloader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=True)
data_loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)
# 获取模型
net = QueryFocusedFullyAttentionNetwork(dataset.concepts).to(training_config.device)
# net.load_state_dict(torch.load("model_para/Epoch.pkl"))
# 定义学习率，损失函数和优化器
learning_rate = 1e-5
criterion = SparseLoss()
l1_regularization = nn.L1Loss()

optimiser = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=rule)


def train():
    net.train()
    best_loss = 20
    test_file = "out/EpochTest.txt"
    if not os.path.exists("out"):
        os.mkdir("out")
    if not os.path.exists("model_para"):
        os.mkdir("model_para")
    for epoch in range(20):
        out_file = "out/Epoch{}.txt".format(epoch)
        print("Epoch {}".format(epoch))
        for step, data in enumerate(train_loader):
            x, y = data
            pred = net.forward(x)
            y = y.to(training_config.device)
            y = torch.squeeze(y)
            loss = criterion(pred, y, 8)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            out_str = "Epoch {} Step {} Loss {}".format(epoch, step, loss.item())
            with open(out_file, mode='a', encoding='utf-8') as out:
                out.write(out_str + "\n")
            print(out_str)
            # print("Epoch {} Step {} Loss {}".format(epoch, step, loss.item()))
        # 在测试集合上的效果
        scheduler.step()
        print("lr of epoch", epoch, "=>", scheduler.get_lr())
        net.eval()
        test_loss = 0
        with torch.no_grad():
            for test_data in test_loader:
                x, y = test_data
                pred = net.forward(x)
                y = y.to(training_config.device)
                y = torch.squeeze(y)
                loss = criterion(pred, y)
                test_loss = test_loss + loss.item()
            mean_loss = test_loss / len(test_dataset) * len(test_loader)
            test_out_str = "Epoch {} Mean Loss {} Best Loss {}".format(epoch, mean_loss, best_loss)
            with open(test_file, mode='a', encoding='utf-8') as out:
                out.write(test_out_str + "\n")
            print(test_out_str)
            if mean_loss < best_loss:
                best_loss = mean_loss
                with open(test_file, mode='a', encoding='utf-8') as out:
                    out.write("EPOCH {} UPDATE MODEL\n".format(epoch))
                # 保存模型参数到路径"./data/model_parameter.pkl"
                torch.save(net.state_dict(), "model_para/Epoch{}.pkl".format(epoch))


def forward_and_optim(data):
    x, y = data
    pred = net.forward(data)
    loss = criterion(pred, torch.tensor(y, device=training_config.device))
    print("优化前loss：{}".format(loss))
    loss.backward()
    print("误差反向传递完成")
    optimiser.step()
    print("梯度优化完成")
    pred = net.forward(data)
    loss = criterion(pred, torch.tensor(y, device=training_config.device))
    print("优化后loss：{}".format(loss))


def forward_and_loss():
    for step, data in enumerate(data_loader):
        with torch.no_grad():
            x, y = data
            print("x{}".format(x))
            pred = net.forward(x)
            y = y.to(training_config.device)
            y = torch.squeeze(y)
            loss = criterion(pred, y)
            print("pred{}".format(pred))
            print("y{}".format(y))
            print("loss.item{}".format(loss.item()))
            break


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


train()
