import threading
import time
from queue import Queue

import cv2 as cv
import pyaudio

from realtime.clock import Clock
from realtime.stream import AudioRecorder, Stream, VideoRecorder


class Player():
    def __init__(self, audio_stream: AudioRecorder, video_stream: Stream):
        self.audio_stream = audio_stream
        self.video_stream = video_stream

        self.audio_queue = Queue()
        self.frame_queue = Queue()

        self.audio_stream.add_consumer(self.audio_queue)
        self.video_stream.add_consumer(self.frame_queue)

        self.pa = pyaudio.PyAudio()
        self.audio_out_stream = self.pa.open(
            rate=self.audio_stream.sr,
            channels=1,
            format=self.audio_stream.format,
            frames_per_buffer=1024,
            output=True,
            start=False
        )

        self._stop = True
        self.ext_clock = Clock()
        self.message_queue = Queue()

    def play(self, audio=True, video=True):
        assert self._stop

        def play_audio():
            self.audio_out_stream.start_stream()

            pts, wave = self.audio_queue.get()
            self.audio_out_stream.write(wave)
            self.ext_clock.set(0.0)
            self.message_queue.put('video_can_play')

            while not self._stop:
                pts, wave = self.audio_queue.get()
                self.audio_out_stream.write(wave)

        def play_video():
            pts, frame = self.frame_queue.get()
            self.message_queue.get()
            while not self._stop:
                next_pts, next_frame = self.frame_queue.get()
                # print(colock.get())
                duration = next_pts - max(self.ext_clock.get(), 0)
                if(duration > 0):
                    cv.cv2.imshow('demo', frame)
                    cv.cv2.waitKey(max(1, int(duration * 1000)))
                pts, frame = next_pts, next_frame

        self._stop = False

        self.audio_play_thread = threading.Thread(target=play_audio, daemon=False)
        self.audio_play_thread.start()
        self.video_play_thread = threading.Thread(target=play_video, daemon=False)
        self.video_play_thread.start()

    def stop(self):
        self._stop = True


if __name__ == "__main__":
    audio_stream = AudioRecorder(160)
    video_stream = VideoRecorder()
    colock = Clock()
    player = Player(audio_stream, video_stream)
    audio_stream.start(colock)
    video_stream.start(colock)
    time.sleep(0.04)
    player.play(audio=True, video=True)
