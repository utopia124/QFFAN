import os
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from config.config import DatasetConfig

dataset_config = DatasetConfig()


def frame_sample(video_path, save_path, frame_rate=15, sample_interval=1):
    """
    帧抽样方法，将给定的视频进行帧抽样
    :param video_path: 视频路径
    :param save_path: 截取帧的保存根路径
    :param frame_rate: 视频帧速率
    :param sample_interval: 取样间隔（秒）
    :return: 帧路径列表
    """
    capture = cv2.VideoCapture(video_path)
    frame_interval = frame_rate * sample_interval
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_idx = 0
    frame_num = 0

    video_name = os.path.basename(video_path).split('.')[0]
    video_frame_save_path = os.path.join(save_path, video_name)
    if not os.path.exists(video_frame_save_path):
        os.mkdir(video_frame_save_path)

    while frame_idx < frame_count:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = capture.read()
        cv2.imwrite(os.path.join(video_frame_save_path, "{}.jpg".format(frame_num)), img)
        frame_num += 1
        frame_idx += frame_interval


def split_tensor(tensor):
    """
    将一个形状为 (n, 1, m) 的张量按照 shot_num_split 切分，并返回切分后的张量列表。
    Args:
        :param tensor:
    Returns:
        切分后的张量列表。
    """
    # 计算切分后每段的数量和最后一段的长度
    tmp = dataset_config.shot_num_split * dataset_config.frames_per_shot
    remainder = tensor.shape[0] % tmp

    # 使用 torch.split() 函数切分张量
    tensor_list = torch.split(tensor[:tensor.shape[0] - remainder], tmp, dim=0)
    tensor_list = list(tensor_list)
    # 如果最后一段的长度不够，则使用全零张量进行 padding
    if remainder > 0:
        zero_tensor = torch.zeros((tmp - remainder, 1, tensor.shape[2]))
        tensor_list.append(torch.cat((tensor[tensor.shape[0] - remainder:], zero_tensor), dim=0))

    return tensor_list


class UTEVideoDataset(Dataset):
    def __init__(self, data_root):
        # 用于提取图像特征的google_net
        self.google_net = GoogleNetFeatureExtractor()
        # 原始数据根目录
        self.data_root = data_root
        # 每镜头包含帧数
        self.frames_per_shot = 5
        # 抽样出来的视频帧的临时保存路径
        self.tmp_save_path = os.path.abspath("tmp")
        # print(self.tmp_save_path)
        # 提取的图像特征tensor list
        self.img_fea = []
        # 每个sample包含的shots数
        self.shot_num_split = dataset_config.shot_num_split
        # 四个视频的路径
        video_p01 = os.path.join(data_root, "P01.mp4")
        video_p02 = os.path.join(data_root, "P02.mp4")
        video_p03 = os.path.join(data_root, "P03.mp4")
        video_p04 = os.path.join(data_root, "P04.mp4")
        self.video_paths = [video_p01, video_p02, video_p03, video_p04]
        # 预处理视频
        self.video_frames = self.read_pt()
        # self.pre_treat_video()
        # 预处理概念字典
        self.concepts = self.pre_treat_concepts()
        # 组织数据
        self.data = self.form_data()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def pre_treat_video(self):
        """
        预处理视频，将视频抽帧并打包
        :return: 包含视频帧路径集合的字典
        """
        # 进行视频帧采样
        if not os.path.exists(self.tmp_save_path):
            print("不存在{}".format(self.tmp_save_path))
            os.mkdir(self.tmp_save_path)
            for path in self.video_paths:
                frame_sample(path, self.tmp_save_path)
        # 将帧路径集合为list
        video_p01_sample = os.path.join(self.tmp_save_path, "P01")
        video_p02_sample = os.path.join(self.tmp_save_path, "P02")
        video_p03_sample = os.path.join(self.tmp_save_path, "P03")
        video_p04_sample = os.path.join(self.tmp_save_path, "P04")
        video_p01_frames = [os.path.join(video_p01_sample, frame) for frame in os.listdir(video_p01_sample)]
        video_p02_frames = [os.path.join(video_p02_sample, frame) for frame in os.listdir(video_p02_sample)]
        video_p03_frames = [os.path.join(video_p03_sample, frame) for frame in os.listdir(video_p03_sample)]
        video_p04_frames = [os.path.join(video_p04_sample, frame) for frame in os.listdir(video_p04_sample)]

        videos = [video_p01_frames, video_p02_frames, video_p03_frames, video_p04_frames]
        for video_frames in videos:
            dir_path = os.path.dirname(video_frames[0])
            feature_path = dir_path + ".pt"
            self.img_fea.append(self.google_net.extract(video_frames, feature_path))

        return {"P01": self.img_fea[0], "P02": self.img_fea[1], "P03": self.img_fea[2], "P04": self.img_fea[3]}

    def read_pt(self):
        videos = ["P01", "P02", "P03", "P04"]
        for video in videos:
            dir_path = os.path.join("tmp", video)
            feature_path = dir_path + ".pt"
            self.img_fea.append(self.google_net.extract([], feature_path))

        return {"P01": self.img_fea[0], "P02": self.img_fea[1], "P03": self.img_fea[2], "P04": self.img_fea[3]}

    def pre_treat_concepts(self):
        """
        预处理概念
        将概念列表读取到内存
        :return: 概念列表
        """
        concepts_path = os.path.join(self.data_root, "Data", "Dense_per_shot_tags", "Dictionary.txt")
        raw = pd.read_csv(concepts_path)
        concepts_raw = np.array(raw).flatten()
        concepts = {concept.replace("'", ""): v for v, concept in enumerate(concepts_raw)}
        return concepts

    def form_data(self):
        """
        组织数据
        由之前的打包好的视频帧集合，概念列表
        通过遍历summary文件，得到dataset的每一个sample
        :return:
        """
        # summary文件的路径
        summary_path = os.path.join(self.data_root, "Data", "Query-Focused_Summaries", "Oracle_Summaries")
        data = []
        # 遍历summary文件
        for root, dir_names, files in os.walk(summary_path):
            # 此判断排除“ReadMe.txt”文件
            if len(files) > 1:
                for file in files:
                    file_path = os.path.join(root, file)
                    # 概念对，即文件名前两个单词
                    concept_pair = file.split("_")[0:2]
                    # 划分好的tensor_list
                    # frames = self.video_frames[os.path.basename(root)]
                    tensor_list = split_tensor(self.video_frames[os.path.basename(root)])
                    # 该视频产生的sample数量
                    sample_num = len(tensor_list)
                    # 该video涉及视频的总镜头数量，向上取整
                    num_shots = sample_num * self.shot_num_split
                    # 建立总的summary向量并赋值
                    summary = np.zeros(num_shots)
                    for i in np.array(pd.read_table(file_path, header=None)).flatten():
                        summary[i - 1] = 1
                    summary = torch.FloatTensor(summary)
                    summary_list = torch.split(summary, self.shot_num_split)
                    # 将数据装填
                    for i in range(sample_num):
                        if sum(summary_list[i]) > 0:
                            data.append([[tensor_list[i], concept_pair], summary_list[i]])
        return data


class GoogleNetFeatureExtractor:
    def __init__(self):
        self.model = models.googlenet(pretrained=True, progress=True).to('cuda')
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image_list, feature_path=None):
        if feature_path is not None and os.path.exists(feature_path):
            features = torch.load(feature_path)
        else:
            print("准备提取特征")
            print(feature_path)
            features = []
            for i, image_path in enumerate(image_list):
                print("\r{}".format(i), end="", flush=True)
                image = Image.open(image_path)
                input_tensor = self.preprocess(image)
                input_tensor = input_tensor.unsqueeze(0)
                input_tensor = input_tensor.to('cuda')
                with torch.no_grad():
                    feature = self.model.forward(input_tensor)
                    # feature = torch.flatten(feature, 1)
                features.append(feature)
            features = torch.stack(features)
            if feature_path is not None:
                torch.save(features, feature_path)
        return features
