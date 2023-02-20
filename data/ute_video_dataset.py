import os
from torch.utils.data import Dataset
import cv2


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


class UTEVideoDataset(Dataset):
    def __init__(self, data_root):
        self.tmp_save_path = "../tmp"
        self.data_root = data_root
        self.pre_treat()

    def __getitem__(self, index):
        return index

    def __len__(self):
        return 100

    def pre_treat(self):
        video_p01 = os.path.join(self.data_root, "P01.mp4")
        video_p02 = os.path.join(self.data_root, "P02.mp4")
        video_p03 = os.path.join(self.data_root, "P03.mp4")
        video_p04 = os.path.join(self.data_root, "P04.mp4")
        video_paths = [video_p01, video_p02, video_p03, video_p04]
        video_frame_sample_path = []
        if not os.path.exists(self.tmp_save_path):
            os.mkdir(self.tmp_save_path)
            for path in video_paths:
                video_frame_sample_path.append(frame_sample(path, self.tmp_save_path))


data_path = "D://Workspace//Data//UTE_video"
ute_dataset = UTEVideoDataset(data_path)
