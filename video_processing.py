import os
import cv2
import numpy as np

class VideoProcessing:
    def __init__(self):
        self.cwd = os.getcwd()
        self.video_dir = os.path.join(self.cwd, 'data\\videos')
        self.videos = {}

    def processing(self):
        args = {
            'width': 540,
            'height': 380
        }
        print('Video processing...')
        for file_name in os.listdir(self.video_dir):

            cap = cv2.VideoCapture(os.path.join(self.video_dir, file_name))
            
            video_frames = []

            print(f'Start processing: {file_name} ({cap.get(cv2.CAP_PROP_FRAME_COUNT)} frames)')

            while (cap.isOpened()):

                ret, frame = cap.read()

                if not ret: 
                    print('Can\'t receive frame (or stream end).')
                    break

                frame = cv2.resize(frame, (args['width'], args['height']), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

                video_frames.append(frame)
            
            self.videos[file_name.replace('.mp4', '')] = np.array(video_frames)
            
            print(f'End processing: {file_name}\n')

    def get_length(self, file_name):
        return cv2.VideoCapture(os.path.join(self.video_dir, f'{file_name}.mp4')).get(cv2.CAP_PROP_FRAME_COUNT)