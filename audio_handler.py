import os
from moviepy.editor import VideoFileClip
import numpy as np

from video_handler import VideoProcessing


class AudioProcessing:
    def __init__(self):
        self.cwd = os.getcwd()
        self.video_dir = os.path.join(self.cwd, 'data\\videos')
        self.sounds = {}

    def process(self, video_id: str, values_per_second: int):
        audio = VideoFileClip(os.path.join(self.video_dir, f'{video_id}.mp4')).audio.to_soundarray()
        audio = audio.mean(axis=1)
        
        frames, fps = VideoProcessing().get_len_fps(video_id)
        target_audio_len = int(frames // (fps // values_per_second))
        
        audio = self.max_min_pooling(audio, len(audio) // frames)
        indices = np.linspace(0, len(audio) - 1, target_audio_len).astype(int)

        return abs(audio[indices])
    
    def max_min_pooling(self, signal, window_size):
        windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)
        max_vals = np.max(windows, axis=1)
        min_vals = np.min(windows, axis=1)
        return (max_vals + min_vals) / 2