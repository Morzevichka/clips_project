import os
import cv2
import numpy as np
from torchvision.transforms import v2
import torch

class VideoProcessing:
    def __init__(self):
        self.cwd = os.getcwd()
        self.video_dir = os.path.join(self.cwd, 'data\\videos')

    def get_video(self, video_id: str, width, height):
        video_path = os.path.join(self.video_dir, f'{video_id}.mp4')

        if not os.path.exists(video_path):
            return
        
        cap = cv2.VideoCapture(video_path)

        frame_total, fps = self.get_len_fps(video_id)
        values_per_second = np.round(fps / (frame_total // 100) * 3.2, 2)

        video_frames = []

        frame_count = 0
        while (cap.isOpened()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (width, height), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            video_frames.append(frame)

            frame_count += (fps // values_per_second)

        video_frames = np.array(video_frames)

        return np.transpose(video_frames, (0, 3, 1, 2)), frame_total

    def get_len_fps(self, video_id):
        frames = int(cv2.VideoCapture(os.path.join(self.video_dir, f'{video_id}.mp4')).get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cv2.VideoCapture(os.path.join(self.video_dir, f'{video_id}.mp4')).get(cv2.CAP_PROP_FPS))
        return frames, fps