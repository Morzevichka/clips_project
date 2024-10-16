from extractor import Extractor
from video_processing import VideoProcessing
from audio_processing import AudioProcessing

# Downlaod videos and get markgers
test = Extractor('data\\video_ids.txt', isFile=True)
test.download()
test.get_markers()

test1 = VideoProcessing()
test1.processing()

