import time
from realtime.stream import MFCCStream, VideoRecorder, DNNVideoStream
from realtime.clock import Clock
from realtime.player import Player


mfcc_stream = MFCCStream()
video_stream = VideoRecorder()
dnn_video_stream = DNNVideoStream(mfcc_stream, video_stream)
colock = Clock()
player = Player(mfcc_stream.audio_stream, dnn_video_stream)
dnn_video_stream.start(colock)
time.sleep(2)
player.play(audio=True, video=True)
