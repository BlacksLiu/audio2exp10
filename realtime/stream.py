import threading
from queue import Queue
import cv2 as cv
import numpy as np
import pyaudio
from python_speech_features import mfcc
from model import Model
from realtime.clock import Clock
import noisereduce as nr
import wave
import librosa

class Stream():
    def __init__(self):
        self.consumers = []
        self._stop = True

    def add_consumer(self, queue):
        self.consumers.append(queue)

    def start(self, clock):
        assert self._stop
        self.thread = threading.Thread(target=self.task(clock), daemon=True)
        self._stop = False
        self.thread.start()

    def task(self, clock):
        raise NotImplementedError

    def stop(self):
        self._stop = True


class VideoRecorder(Stream):

    def __init__(self, device=0):
        super(VideoRecorder, self).__init__()
        self.cap = cv.cv2.VideoCapture(device)

    def task(self, clock):
        def grab_frame():
            while not self._stop:
                ret, frame = self.cap.read()
                pts = clock.get()
                if not ret or pts < 0:
                    continue
                for consumer in self.consumers:
                    consumer.put((pts, frame))
        return grab_frame


class AudioRecorder(Stream):

    def __init__(self, chunk, sr=16000, format=pyaudio.paInt16):
        super(AudioRecorder, self).__init__()
        self.chunk = chunk
        self.sr = sr
        self.format = format
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(rate=sr, channels=1, format=format, frames_per_buffer=chunk, input=True, start=False)
        self.count = 0
        self.noise_wf = wave.open('data/noise.wav', 'rb')
        self.noise = self.noise_wf.readframes(4 * sr)

    def task(self, clock: Clock):
        def grab_frame():
            self.stream.start_stream()
            while not self._stop:
                wave = self.stream.read(self.chunk)
                pts = self.get_pts()
                clock.set(pts)
                for consumer in self.consumers:
                    consumer.put((pts, wave))
        return grab_frame

    def stop(self):
        self._stop = True
        self.count = 0
        self.stream.stop_stream()

    def get_pts(self):
        pts = self.count * self.chunk / self.sr
        self.count += 1
        return pts


class MFCCStream(Stream):
    def __init__(self, sr=16000, winlen=0.025, winstep=0.01, numcep=26, nfft=1024):
        super(MFCCStream, self).__init__()
        self.sr = sr
        self.winlen = winlen
        self.winstep = winstep
        self.numcep = numcep
        self.nfft = nfft
        self.audio_stream = AudioRecorder(int(winstep * sr), sr=sr)
        self.audio_read_queue = Queue()
        self.audio_stream.add_consumer(self.audio_read_queue)
        # self.noise_wf = wave.open('data/noise.wav', 'rb')
        # self.noise = self.noise_wf.readframes(4 * sr)
        # self.noise = np.frombuffer(self.noise, np.int16)

    def task(self, clock):
        def generate_mfcc():
            self.audio_stream.start(clock)
            history = np.zeros(int(self.sr * (self.winlen - self.winstep)), dtype=np.float32)
            while not self._stop:
                pts, wave = self.audio_read_queue.get()
                # samples = np.frombuffer(wave, np.int16)
                samples = librosa.util.buf_to_float(wave)
                # samples = nr.reduce_noise(samples.astype(np.float32), self.noise.astype(np.float32))
                mfcc_window = np.concatenate([history, samples])
                history = mfcc_window[int(self.sr * self.winstep):]
                # mfcc_window = librosa.util.normalize(mfcc_window, norm=np.inf, axis=None)

                cur_mfcc = mfcc(mfcc_window, samplerate=self.sr,
                                winlen=self.winlen, winstep=self.winstep,
                                numcep=self.numcep, nfft=self.nfft)
                cur_mfcc = cur_mfcc[0, :]
                for consumer in self.consumers:
                    consumer.put((pts, cur_mfcc))
        return generate_mfcc

    def stop(self):
        self.audio_stream.stop()
        self._stop = True


class DNNVideoStream(Stream):
    def __init__(self, mfcc_stream: MFCCStream, video_stream: VideoRecorder, num_context=9):
        super(DNNVideoStream, self).__init__()
        self.mfcc_stream = mfcc_stream
        self.video_stream = video_stream
        self.num_context = num_context
        self.mfcc_queue = Queue()
        self.frame_queue = Queue()
        self.mfcc_stream.add_consumer(self.mfcc_queue)
        self.video_stream.add_consumer(self.frame_queue)
        self.model = Model(9, self.mfcc_stream.numcep)

    def task(self, clock: Clock):
        def process_frame():
            self.mfcc_stream.start(clock)
            self.video_stream.start(clock)
            valid_window = 0.015

            while not self._stop:
                # get a frame
                frame_pts, frame = self.frame_queue.get()
                while True:
                    # update mfcc history
                    mfcc_pts, mfcc = self.mfcc_queue.get()
                    self.model.update_mfcc(mfcc)
                    # get the pts from which we can predict a alpha_exp coeff
                    record_clock = clock.get()
                    pts_start = record_clock - self.num_context * self.mfcc_stream.winstep

                    if pts_start < frame_pts - valid_window:
                        # just continue to get next mfcc
                        continue
                    elif pts_start > frame_pts + valid_window:
                        # just break and get next frame
                        break

                    # get current mfcc window
                    while mfcc_pts < record_clock:
                        # update mfcc history
                        mfcc_pts, mfcc = self.mfcc_queue.get()
                        self.model.update_mfcc(mfcc)
                    # do mask clip
                    frame = self.model.render(frame)
                    # put in comsumer queue
                    for consumer in self.consumers:
                        consumer.put((frame_pts, frame))

        return process_frame

    def stop(self):
        self._stop = True
        self.mfcc_stream.stop()
        self.video_stream.stop()


if __name__ == "__main__":
    # video_recoder = VideoRecorder()
    # clock = Clock()
    # queue = Queue()
    # clock.set(0.)
    # video_recoder.add_consumer(queue)
    # video_recoder.start(clock)
    # while True:
    #     pts, frame = queue.get()
    #     print(pts)
    #     cv.cv2.imshow('video_recorder', frame)
    #     cv.cv2.waitKey(1) 

    # audio_stream = AudioRecorder(160)
    # clock = Clock()
    # queue = Queue()
    # audio_stream.add_consumer(queue)
    # audio_stream.start(clock)
    # while True:
    #     pts, wave = queue.get()
    #     print(pts)

    # mfcc_stream = MFCCStream()
    # clock = Clock()
    # queue = Queue()
    # mfcc_stream.add_consumer(queue)
    # mfcc_stream.start(clock)
    # while True:
    #     pts, cur_mfcc = queue.get()
    #     print(pts, cur_mfcc.shape)

    mfcc_stream = MFCCStream()
    video_stream = VideoRecorder()
    dnn_video_stream = DNNVideoStream(mfcc_stream, video_stream)
    clock = Clock()
    queue = Queue()
    dnn_video_stream.add_consumer(queue)
    dnn_video_stream.start(clock)
    while True:
        pts, frame = queue.get()
        print(pts, frame.shape)
