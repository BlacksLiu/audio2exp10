import pyaudio
import librosa
import numpy as np
import threading
import time
from python_speech_features import mfcc


class WindowedMfccStream():
    PYAUDIO_FORMAT = pyaudio.paInt16
    CHANNELS = 1

    def __init__(self, sr=16000, win_length=0.025, win_step=0.01,
                 num_cep=26, nfft=1024, prev_num_context=9,
                 next_num_context=9, out_num_frames=16, fps=25):
        self.sr = sr
        self.win_length = 0.025
        self.win_step = 0.01
        self.num_cep = num_cep
        self.nfft = nfft
        self.prev_num_context = prev_num_context
        self.next_num_context = next_num_context
        self.out_num_frames = out_num_frames
        self.fps = fps
        assert self.win_length > self.win_step

        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=WindowedMfccStream.PYAUDIO_FORMAT,
                                   channels=WindowedMfccStream.CHANNELS,
                                   rate=self.sr, input=True, output=True, start=False, frames_per_buffer=16000)
        self._stop = True

    def init_mfccs(self):
        num_empty_mfccs = self.prev_num_context + (self.out_num_frames - 1) * int((1 / self.fps) / self.win_step)
        self.mfccs = [np.zeros((1, self.num_cep)) for i in range(num_empty_mfccs)]
        self.out_num_mfcc = num_empty_mfccs + self.next_num_context + 1

    def start(self):

        def record_mfcc():
            self.init_mfccs()
            self._stop = False
            last_samples = self.stream.read(int(self.win_length * self.sr))
            last_samples = np.frombuffer(last_samples, np.int16).astype(np.float32)
            self.mfccs.append(
                mfcc(last_samples, samplerate=self.sr, winlen=self.win_length, winstep=self.win_step,
                     numcep=self.num_cep, nfft=self.nfft)[:1, :]
            )
            next_start = int(self.win_step * self.sr)
            while not self._stop:
                samples = self.stream.read(int(self.win_step * self.sr))
                self.stream.write(samples)
                samples = np.frombuffer(samples, dtype=np.int16).astype(np.float32)
                samples = np.concatenate([last_samples[next_start:], samples], axis=0)
                self.mfccs.append(
                    mfcc(samples, samplerate=self.sr, winlen=self.win_length, winstep=self.win_step,
                         numcep=self.num_cep, nfft=self.nfft)[:1, :]
                )
        self._record_thread = threading.Thread(target=record_mfcc, daemon=True)
        self.time = time.time()
        self.stream.start_stream()
        self._record_thread.start()
        self._stop = False
        return self.time

    def stop(self):
        self._stop = True
        self.stream.stop_stream()

    def clear(self):
        assert self._stop
        self.init_mfccs()

    def get(self, timestamp):
        assert not self._stop and self.time < timestamp
        # print(timestamp - self.time)
        start = int((timestamp - self.time) / self.win_step)
        end = start + self.out_num_mfcc
        # print(start, end)
        while end > len(self.mfccs):
            time.sleep(self.win_step)
        mfccs_for_per_frame = []
        step = int((1 / self.fps) / self.win_step)

        for i in range(self.out_num_frames):
            middle = start + self.prev_num_context + (step * i)
            mfccs_for_per_frame.append(
                np.concatenate(self.mfccs[middle - self.prev_num_context:middle + self.next_num_context + 1]))

        mfccs_for_per_frame = np.stack(mfccs_for_per_frame, axis=0)
        return mfccs_for_per_frame


class WindowedAudioFileMfccStream():
    PYAUDIO_FORMAT = pyaudio.paInt16
    CHANNELS = 1

    def __init__(self, f, sr=16000, win_length=0.025, win_step=0.01,
                 num_cep=26, nfft=1024, prev_num_context=9,
                 next_num_context=9, out_num_frames=16, fps=25):
        self.f = f
        self.sr = sr
        self.win_length = 0.025
        self.win_step = 0.01
        self.num_cep = num_cep
        self.nfft = nfft
        self.prev_num_context = prev_num_context
        self.next_num_context = next_num_context
        self.out_num_frames = out_num_frames
        self.fps = fps
        assert self.win_length > self.win_step

        self._stop = True

    def init_mfccs(self):
        num_empty_mfccs = self.prev_num_context + (self.out_num_frames - 1) * int((1 / self.fps) / self.win_step)
        # self.mfccs = [np.zeros((1, self.num_cep)) for i in range(num_empty_mfccs)]
        samples, _ = librosa.load(self.f, sr=self.sr)
        print('extracting mfcc ...')
        mfccs = mfcc(samples, samplerate=self.sr, winlen=self.win_length, winstep=self.win_step,
                     numcep=self.num_cep, nfft=self.nfft)
        print('extracting mfcc ok.')
        self.mfccs = [mfccs[i:i+1, :] for i in range(mfccs.shape[0])]         
        self.out_num_mfcc = num_empty_mfccs + self.next_num_context + 1

    def start(self):
        self.init_mfccs()
        self.time = time.time()
        self._stop = False
        return self.time

    def stop(self):
        self._stop = True
        self.stream.stop_stream()

    def clear(self):
        assert self._stop
        self.init_mfccs()

    def get(self, timestamp):
        assert not self._stop and self.time < timestamp
        print(timestamp - self.time)
        start = int((timestamp - self.time) / self.win_step)
        end = start + self.out_num_mfcc
        print(start, end)
        if end > len(self.mfccs):
            raise IndexError()
        mfccs_for_per_frame = []
        step = int((1 / self.fps) / self.win_step)

        for i in range(self.out_num_frames):
            middle = start + self.prev_num_context + (step * i)
            mfccs_for_per_frame.append(
                np.concatenate(self.mfccs[middle - self.prev_num_context:middle + self.next_num_context + 1]))

        mfccs_for_per_frame = np.stack(mfccs_for_per_frame, axis=0)
        return mfccs_for_per_frame


if __name__ == "__main__":
    stream = WindowedMfccStream(prev_num_context=9, next_num_context=0)
    t = stream.start()
    unit = 0.04

    for i in range(1, 100):
        # get a frame
        time.sleep(0.03)
        s = time.time()
        # exit(0)
        stream.get(t + unit * i)
        print((time.time() - t) / unit)
        print(time.time() - s)


        # ss = time.time()
        # stream.get(s)
        # t += unit
        # print(time.time() - ss)
        # print(stream.get(t))
        # print(time.time() - ss)
        # print(stream.read(160))
        # pass
