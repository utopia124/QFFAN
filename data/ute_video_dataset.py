import os
from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np


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


class UTEVideoDataset(Dataset):
    def __init__(self, data_root):
        # 原始数据根目录
        self.data_root = data_root
        # 每镜头包含帧数
        self.frames_per_shot = 5
        # 抽样出来的视频帧的临时保存路径
        self.tmp_save_path = os.path.abspath("tmp")
        # print(self.tmp_save_path)
        # 四个视频的路径
        video_p01 = os.path.join(data_root, "P01.mp4")
        video_p02 = os.path.join(data_root, "P02.mp4")
        video_p03 = os.path.join(data_root, "P03.mp4")
        video_p04 = os.path.join(data_root, "P04.mp4")
        self.video_paths = [video_p01, video_p02, video_p03, video_p04]
        # 预处理视频
        self.video_frames = self.pre_treat_video()
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
        return {"P01": video_p01_frames, "P02": video_p02_frames, "P03": video_p03_frames, "P04": video_p04_frames}

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
                    # 打包好的帧集合
                    frames = self.video_frames[os.path.basename(root)]
                    # 该sample涉及视频的镜头数量，向上取整
                    num_shots = len(frames) // self.frames_per_shot + 1
                    # 处理好的summary向量
                    summary = np.zeros(num_shots)
                    for i in np.array(pd.read_table(file_path, header=None)).flatten():
                        summary[i - 1] = 1
                    # 将数据组装成一个sample，并入总数据
                    data.append(((frames, concept_pair), summary))
        return data

# data_path = "D://Workspace//Data//UTE_video"
# ute_dataset = UTEVideoDataset(data_path)
#
# print(len(ute_dataset[10][0][0]))
