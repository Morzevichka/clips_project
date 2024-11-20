import os
from moviepy.editor import AudioFileClip
import numpy as np

from media_processing.video_handler import VideoProcessing

class AudioProcessing:
    def __init__(self):
        self.cwd = os.getcwd()
        self.video_dir = os.path.join(self.cwd, 'data\\videos')

    def get_audio(self, video_id: str, target_len: int, train=False):
        audio = AudioFileClip(os.path.join(self.video_dir, f'{video_id}.mp4')).to_soundarray(fps=16000)
        audio = audio.mean(axis=1)
        
        frames, fps = VideoProcessing().get_len_fps(video_id)
        
        audio = self.max_min_pooling(audio, len(audio) // frames)
        indices = np.linspace(0, len(audio) - 1, target_len).astype(int)
        audio = abs(audio[indices])
        return audio.reshape(-1, 1)
    
    def max_min_pooling(self, signal, window_size):
        windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)
        max_vals = np.max(windows, axis=1)
        min_vals = np.min(windows, axis=1)
        return (max_vals + min_vals) / 2
