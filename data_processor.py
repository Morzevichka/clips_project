from extractor import Extractor
from video_handler import VideoProcessing
from audio_handler import AudioProcessing
from database_connection import DatabaseConnection
import os
import numpy as np
from tqdm import tqdm

class DataProcessor:
    def __init__(self, 
                 file: str = 'data\\video_ids.txt', 
                 isFile=True, 
                 db_name: str = 'clip_db', 
                 table_name: str = 'video_dataset'):
        self.extractor = Extractor()
        self.dbconnector = DatabaseConnection(db_name, table_name)
        self.dbconnector.connect()
        self.audio_processor = AudioProcessing()
        self.video_processor = VideoProcessing()

        self.video_ids = []
        if isFile:
            self.video_ids = self._get_ids_from_file(file)
        else:
            self.video_ids = file

    def process_train_test(self, 
                     values_per_second: float = 2.0, 
                     width: int = 640, 
                     height: int = 360,
                     resolution: str = '360p'):

        for video_id in tqdm(self.video_ids, desc='Processing Videos', unit='video'):
            if self.dbconnector.ifExists(video_id) >= 1:
                continue

            self.extractor.download(video_id, resolution=resolution)
            
            frames, fps = self.video_processor.get_len_fps(video_id)
            values_per_second = np.round(fps / (frames // 100) * 3.2, 4)

            markers = self.extractor.get_markers(video_id, values_per_second)
            video_array = self.video_processor.process(video_id, values_per_second, width=width, height=height)
            audio_array = self.audio_processor.process(video_id, values_per_second)

            self.dbconnector.execute_query(video_id, markers, video_array, audio_array)
        self.extractor.driver.quit()

    def process_to_model(self):
        for video_id in self.video_ids:
            yield self.dbconnector.fetch_data(video_id)

    def _get_ids_from_file(self, file: str):
        path = os.path.join(os.getcwd(), file)
        try:
            with open(path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f'File {file} was not found')
            return