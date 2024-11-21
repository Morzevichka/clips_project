from media_processing.video_handler import VideoProcessing
from media_processing.audio_handler import AudioProcessing

import torch
from moviepy.video.io.VideoFileClip import VideoFileClip
from model import Net_v2_Improved
import os
import numpy as np
import matplotlib.pyplot as plt 

class VideoTrimmer:
    def __init__(self, videos_folder, output_folder, model_name):
        self.videos_folder = videos_folder
        self.output_folder = output_folder
        self.model = self.load_model(model_name)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def load_model(self, model_name):
        model = Net_v2_Improved()
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'data', 'models', f'{model_name}.pth'), weights_only=True))
        return model 
        
    def predict(self, video, audio):
        model, video, audio = self.model.to(self.device), video.to(self.device), audio.to(self.device)
        model.eval()
        with torch.no_grad():
            pred = (model(video, audio) > 0.5).float().view(-1).cpu().numpy()
            pred_m, pred_c = model(video, audio).cpu().numpy(), pred
            pred_plots(pred_m, pred_c)
        return pred
    
    def extend_video(self, pred, frames):
        x_original = np.linspace(0, 1, len(pred))
        x_target = np.linspace(0, 1, frames)

        interpolated = np.interp(x_target, x_original, pred)
        stretched = (interpolated > 0.5).astype(int)
        return stretched
    
    def find_interesting_moments(self, video):
        changes = np.diff(np.concatenate(([0], video, [0]))) 
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0] - 1

        sequences = np.column_stack((starts, ends))
        sequences = sorted(sequences, key=lambda x: x[1] - x[0], reverse=True)
        return np.array(sequences)
    
    def trim_video(self, video_name, video_path, intervals, num_clips, target_len=20):
        intervals = intervals[:num_clips]

        video = VideoFileClip(video_path)
        fps = video.fps
        len_video = video.duration

        for i, (start, end) in enumerate(intervals):
            start, end = start / fps, end / fps
            duration_clip = end - start
            if duration_clip < target_len:
                if end > len_video * 0.9:
                    start -= (target_len - duration_clip) * 2/3
                elif start < len_video * 0.1:
                    start = 0 
                    end += (target_len - duration_clip) * 1/3
                else:
                    start -= (target_len - duration_clip) * 2/3
                    end += (target_len - duration_clip) * 1/3

            trimmed = video.subclip(start, end)
            output_path = os.path.join(self.output_folder, f'{video_name}_{start:.2f}-{end:.2f}.mp4')
            trimmed.write_videofile(output_path, codec='libx264')
        
        video.close()
    
    def predict_trimmed(self, video_name, clips, rec_clip_duration=10):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        video_path = os.path.join(self.videos_folder, f'{video_name}.mp4')
        video, frames = VideoProcessing().get_video(video_name, height=96, width=96)
        audio = AudioProcessing().get_audio(video_name, video.shape[0])

        prediction = self.predict(torch.Tensor(video), torch.Tensor(audio))

        video_predictions = self.extend_video(prediction, frames)

        interesting_moments = self.find_interesting_moments(video_predictions)

        self.trim_video(video_name, video_path, interesting_moments, clips, target_len=rec_clip_duration)

def pred_plots(pred_unclass, pred_class):
    pred = [pred_unclass, pred_class]
    title = ['Before classification', 'After classification']
    fig, ax = plt.subplots(1, 2, figsize=(7,4))
    for col in range(2):
        ax[col].plot(range(len(pred[col])), pred[col])
        ax[col].set_title(title[col])
        ax[col].set_yticks([0, 1])
        ax[col].set_xticks([])
    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    plt.show()