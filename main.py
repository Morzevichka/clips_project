from extractor import Extractor
from video_processing import VideoProcessing
from audio_processing import AudioProcessing

# Downlaod videos and get markgers
extractor = Extractor('data\\video_ids.txt', isFile=True)
extractor.download()
extractor.get_markers()

video = VideoProcessing()
video.processing()

audio = AudioProcessing()
audio.processing()