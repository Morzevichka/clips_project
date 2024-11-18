from media_processing.extractor import Extractor
from media_processing.video_handler import VideoProcessing
from media_processing.audio_handler import AudioProcessing

import os
import numpy as np
from tqdm import tqdm
import h5py
import torch
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, 
                 file: str = 'data\\video_ids.txt'):
        self.extractor = Extractor()
        self.audio_processor = AudioProcessing()
        self.video_processor = VideoProcessing()
        self.file = file

        if self.file[-4:] == '.txt':
            self.video_ids = self._get_ids_from_file(self.file)
        else:
            self.video_ids = np.array(self.file)

    def load_dataset(self,
                            values_per_second: float = 2.0, 
                            width: int = 144, 
                            height: int = 144,
                            resolution: str = '360p'):
        with h5py.File('data\\video_audio_data.h5', 'a') as f:
            for video_id in tqdm(self.video_ids, desc='Processing Videos', unit='video'):
                try:
                    _ = f[f'video_{video_id}']
                except KeyError:
                    self.extractor.download(video_id, resolution=resolution)
                
                    frames, fps = self.video_processor.get_len_fps(video_id)
                    values_per_second = np.round(fps / (frames // 100) * 3.2, 2)

                    marker = self.extractor.get_markers(video_id, values_per_second)
                    marker = self.extractor.preprocessing_markers(marker)
                    video = self.video_processor.get_video(video_id, values_per_second, width=width, height=height)
                    audio = self.audio_processor.get_audio(video_id, values_per_second)

                    video_group = f.create_group(f'video_{video_id}')

                    video_dataset = video_group.create_dataset('frames',
                                                                shape=(0, 3, height, width),
                                                                maxshape=(None, 3, height, width),
                                                                dtype='float32',
                                                                compression='gzip',
                                                                compression_opts=9)
                    audio_dataset = video_group.create_dataset('audio', 
                                                                shape=(0, 1),
                                                                maxshape=(None, 1),
                                                                dtype='float32',
                                                                compression='gzip',
                                                                compression_opts=9)
                    marker_dataset = video_group.create_dataset('marker',
                                                                shape=(0, 1), 
                                                                maxshape=(None, 1),
                                                                dtype='float32',
                                                                compression='gzip',
                                                                compression_opts=9)
                    
                    compressed_size = video.shape[0]

                    video_dataset.resize((video_dataset.shape[0] + compressed_size, 3, height, width))
                    video_dataset[-compressed_size:] = video

                    audio_dataset.resize((audio_dataset.shape[0] + compressed_size, 1))
                    audio_dataset[-compressed_size:] = audio

                    marker_dataset.resize((marker_dataset.shape[0] + compressed_size, 1))
                    marker_dataset[-compressed_size:] = marker

    def data_for_model(self, shuffle: bool = True):
        with h5py.File('data\\video_audio_data.h5', 'r') as f:
            if shuffle:
                p = np.random.permutation(range(len(self.video_ids)))
            else:
                p = list(range(len(self.video_ids)))
            for video_id in self.video_ids[p]:
                video_group = f[f'video_{video_id}']

                video = torch.tensor(video_group['frames'][:], dtype=torch.float32)
                audio = torch.tensor(video_group['audio'][:], dtype=torch.float32)
                marker = torch.tensor(video_group['marker'][:], dtype=torch.float32)
                yield marker, video, audio, video_id

    def _get_ids_from_file(self, file: str):
        path = os.path.join(os.getcwd(), file)
        try:
            with open(path, 'r') as f:
                return np.array([line.strip() for line in f.readlines()])
        except FileNotFoundError:
            print(f'File {file} was not found')
            return
        
class DataLoader:
    def __init__(self, shuffle = True):
        self.file_path = os.path.join(os.getcwd(), 'data', 'video_audio_data.h5')
        self.shuffle = shuffle
        self.video_ids = DataProcessor().video_ids
        
    def train(self):
        p = np.random.permutation(range(len(self.train_videos)))
        with h5py.File(self.file_path, 'r') as f:
            for video_id in self.train_videos[p]:
                video_group = f[f'video_{video_id}']

                video = torch.tensor(video_group['frames'][:], dtype=torch.float32)
                audio = torch.tensor(video_group['audio'][:], dtype=torch.float32)
                marker = torch.tensor(video_group['marker'][:], dtype=torch.float32)
                yield marker, video, audio, video_id
            
    def test(self):
        with h5py.File(self.file_path, 'r') as f:
            for video_id in self.val_videos:
                video_group = f[f'video_{video_id}']

                video = torch.tensor(video_group['frames'][:], dtype=torch.float32)
                audio = torch.tensor(video_group['audio'][:], dtype=torch.float32)
                marker = torch.tensor(video_group['marker'][:], dtype=torch.float32)
                yield marker, video, audio, video_id
    
    def train_test_datasets(self, train_size=0.8):
        self.train_videos, self.val_videos = train_test_split(self.video_ids, train_size=train_size, shuffle=self.shuffle)