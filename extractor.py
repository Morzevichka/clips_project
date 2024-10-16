from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By

import os
import numpy as np 
from typing import Any
from pytubefix import YouTube
from video_processing import VideoProcessing
import matplotlib.pyplot as plt

class Extractor:
    def __init__(self, file: Any, isFile: bool = True):
        self.video_ids = []
        self.cwd = os.getcwd()
        self.base_yt_url = 'https://www.youtube.com/watch?v='
        self._configure_webdriver()
        self.video_markers = {}

        if isFile:
            self._get_ids_from_file(file)
        else:
            self.video_ids = file

    def _configure_webdriver(self):
        self.options = Options()
        self.options.add_argument('--headless')
        self.options.add_argument('--log-level=3')
        self.driver = webdriver.Chrome(options=self.options)

    def get_markers(self):
        print('Getting markers...')

        for video_id in self.video_ids:
            self.driver.get(self.base_yt_url + video_id)

            heatmap_svg = WebDriverWait(self.driver, 10)\
                .until(expected_conditions.presence_of_element_located((By.CLASS_NAME, 'ytp-heat-map-chapter')))\
                .get_attribute('innerHTML')
            
            soup = BeautifulSoup(heatmap_svg, 'html.parser')
            heatmap_raw = soup.find('path', attrs='ytp-heat-map-path').get('d')
 
            coordinates = heatmap_raw.replace('M ', '').replace('C ', '').split(' ')[1:]
            points = []
            for c in coordinates:
                x_y = c.split(',')
                points.append([float(x_y[0]), float(x_y[1])])

            markers = np.array(points)[:, 1]
            markers = 90.0 - markers
            markers = np.where(markers > 0.0, markers, 0.0)
            markers = np.where(markers > 100.0, 100, markers)

            video_len = int(VideoProcessing().get_length(video_id))

            marker_indices = np.linspace(0, video_len - 1, num=markers.size).astype(int)

            self.video_markers[video_id] = np.interp(np.arange(video_len), marker_indices, markers)

            print(f'Completed: {video_id}')

        self.driver.quit()
        return self
                

    def download(self, resolution: str ='360p'):
        path = os.path.join(self.cwd, 'data\\videos')

        if not os.path.exists(path):
            os.makedirs(path)

        print('Video downloading...')
        for video_id in self.video_ids:
            if os.path.exists(os.path.join(path, f'{video_id}.mp4')):
                print(f'Video {video_id} is already downloaded!')
                continue

            url = self.base_yt_url + video_id
            yt = YouTube(url)

            print(f'Downloading video: {yt.title}...')
            ys = yt.streams.filter(progressive=True, file_extension='mp4', resolution=resolution).first()
            ys.download(output_path=path, filename=f'{video_id}.mp4')
            print(f'Downloaded: {yt.title}')

        return self
            
    def _get_ids_from_file(self, file: str):
        path = os.path.join(self.cwd, file)
        try:
            with open(path, 'r') as f:
                self.video_ids = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f'File {file} was not found')
            return
        return self