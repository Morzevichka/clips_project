import os
from moviepy.editor import VideoFileClip
import numpy as np
from video_processing import VideoProcessing


class AudioProcessing:
    def __init__(self):
        self.cwd = os.getcwd()
        self.video_dir = os.path.join(self.cwd, 'data\\videos')
        self.sounds = {}

    def processing(self):
        for file_name in os.listdir(self.video_dir):
            audio = VideoFileClip(os.path.join(self.video_dir, file_name)).audio.to_soundarray()
            audio = audio.mean(axis=1)
            
            video_len = int(VideoProcessing().get_length(file_name.replace('.mp4', '')))
            
            factor = len(audio) // video_len
            window_size = 5
            audio = np.convolve(audio[:factor*video_len], np.ones(window_size)/window_size, mode='valid')

            indices = np.linspace(0, len(audio) - 1, video_len).astype(int)

            self.sounds[file_name.replace('.mp4', '')] = audio[indices]