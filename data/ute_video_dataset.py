import os
from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np


def frame_sample(video_path, save_path, frame_rate=15, sample_interval=1):
    """
    :param video_path: 视频路径
    :param save_path: 截取帧的保存根路径
    :param frame_rate: 视频帧速率
    :param sample_interval: 取样间隔（秒）
    :return: 帧路径列表
    """
    capture = cv2.VideoCapture(video_path)
    frame_list = []
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
    return video_frame_save_path


class UTEVideoDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.frames_per_shot = 5
        self.video_p01 = os.path.join(data_root, "P01.mp4")
        self.video_p02 = os.path.join(data_root, "P02.mp4")
        self.video_p03 = os.path.join(data_root, "P03.mp4")
        self.video_p04 = os.path.join(data_root, "P04.mp4")

        self.tmp_save_path = os.path.abspath("../tmp")
        self.video_paths = [self.video_p01, self.video_p02, self.video_p03, self.video_p04]

        # 预处理视频
        self.video_frames = self.pre_treat_video()
        # 预处理概念字典
        self.concepts = self.pre_treat_concepts()

        self.data = self.form_data()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def pre_treat_video(self):
        video_frame_save_paths = []
        if not os.path.exists(self.tmp_save_path):
            os.mkdir(self.tmp_save_path)
            for path in self.video_paths:
                video_frame_save_path = frame_sample(path, self.tmp_save_path)
                video_frame_save_paths.append(video_frame_save_path)
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
        concepts_path = os.path.join(self.data_root, "Data", "Dense_per_shot_tags", "Dictionary.txt")
        raw = pd.read_csv(concepts_path)
        concepts_raw = np.array(raw).flatten()
        concepts = [concept.replace("'", "") for concept in concepts_raw]
        return concepts

    def form_data(self):
        summary_path = os.path.join(self.data_root, "Data", "Query-Focused_Summaries", "Oracle_Summaries")
        data = []
        for root, dir_names, files in os.walk(summary_path):
            if len(files) > 1:
                for file in files:
                    file_path = os.path.join(root, file)
                    concept_pair = file.split("_")[0:2]
                    frames = self.video_frames[os.path.basename(root)]
                    num_shots = len(frames) // self.frames_per_shot + 1
                    summary = np.zeros(num_shots)
                    for i in np.array(pd.read_table(file_path, header=None)).flatten():
                        summary[i - 1] = 1
                    data.append(((frames, concept_pair), summary))
        return data


data_path = "D://Workspace//Data//UTE_video"
ute_dataset = UTEVideoDataset(data_path)

print(ute_dataset[10])


