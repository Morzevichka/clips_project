from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By

import os
import numpy as np 
from pytubefix import YouTube
import matplotlib.pyplot as plt

from video_handler import VideoProcessing

class Extractor:
    def __init__(self):
        self.cwd = os.getcwd()
        self.base_yt_url = 'https://www.youtube.com/watch?v='
        self._configure_webdriver()

    def _configure_webdriver(self):
        self.options = Options()
        self.options.add_argument('--headless')
        self.options.add_argument('--log-level=3')
        self.options.add_argument("--mute-audio")
        self.driver = webdriver.Chrome(options=self.options)

    def get_markers(self, video_id: str, values_per_second: int):
        frames, fps = VideoProcessing().get_len_fps(video_id)

        self.driver.get(self.base_yt_url + video_id)
        heatmap_svg = WebDriverWait(self.driver, 10)\
            .until(expected_conditions.presence_of_element_located((By.CLASS_NAME, 'ytp-heat-map-chapter')))\
            .get_attribute('innerHTML')

        soup = BeautifulSoup(heatmap_svg, 'html.parser')
        heatmap_raw = soup.find('path', attrs='ytp-heat-map-path').get('d')

        coordinates_raw = heatmap_raw.replace('M ', '').replace('C ', '').split(' ')[1:]
        coordinates = []

        for c in coordinates_raw:
            x_y = c.split(',')
            coordinates.append([float(x_y[0]), float(x_y[1])])

        markers = np.array(coordinates)[:, 1]
        markers = 90.0 - markers
        markers = np.where(markers > 0.0, markers, 0.0)
        markers = np.where(markers > 100.0, 100, markers)


        marker_indices = np.linspace(0, frames // (fps // values_per_second) - 1, num=markers.size).astype(int)
        markers = np.interp(np.arange(frames // (fps // values_per_second)), marker_indices, markers)

        return markers
                
    def download(self, video_id: str, resolution: str ='360p'):
        path = os.path.join(self.cwd, 'data\\videos')

        if not os.path.exists(path):
            os.makedirs(path)

        if os.path.exists(os.path.join(path, f'{video_id}.mp4')):
            return

        url = self.base_yt_url + video_id
        yt = YouTube(url)
        ys = yt.streams.filter(progressive=True, file_extension='mp4', resolution=resolution).first()
        ys.download(output_path=path, filename=f'{video_id}.mp4')
        return
    
    def markers_plot(self, video_id: str, markers: np.ndarray):
        plt.plot(range(markers.size), markers)
        plt.title(f'The heatmap of {video_id}')
        plt.xlabel('Frames')
        plt.ylabel('Level of Intensity')
        plt.show()
        return