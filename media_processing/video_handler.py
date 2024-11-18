import os
import cv2
import numpy as np

class VideoProcessing:
    def __init__(self):
        self.cwd = os.getcwd()
        self.video_dir = os.path.join(self.cwd, 'data\\videos')

    def get_video(self, video_id: str, values_per_second: int, width: int = 640, height: int = 360):
        video_path = os.path.join(self.video_dir, f'{video_id}.mp4')

        if not os.path.exists(video_path):
            return
        
        cap = cv2.VideoCapture(video_path)
        _, fps = self.get_len_fps(video_id)

        video_frames = []

        frame_count = 1
        while (cap.isOpened()):
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % (fps // values_per_second) == 0:
                frame = cv2.resize(frame, (width, height), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                video_frames.append(frame)
                (144,144,3)
            frame_count += 1
        
        if len(video_frames) == cap.get(cv2.CAP_PROP_FRAME_COUNT) // (fps // values_per_second):
            frames = np.array(video_frames)
            return np.transpose(frames, (0, 3, 1, 2))
        else:
            print('Error during video...')

    def get_len_fps(self, video_id):
        frames = int(cv2.VideoCapture(os.path.join(self.video_dir, f'{video_id}.mp4')).get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cv2.VideoCapture(os.path.join(self.video_dir, f'{video_id}.mp4')).get(cv2.CAP_PROP_FPS))
        return frames, fps
