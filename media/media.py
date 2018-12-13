import os
import pickle
import subprocess
from abc import ABC, abstractmethod

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from imutils.video import FileVideoStream, FPS
from numpy.core.multiarray import dtype
from scipy.io import wavfile

from synchronize import get_directogram, get_impact_envelopes, get_onset_envelopes


class Media(ABC):
    def __init__(self, file_name):
        self.file_name = file_name
        self.convert_file()
        self.data_name = self.file_name + '_data.pkl'
        self.sample_rate = None
        self.time_series = None
        self.envelope_times = None
        self.envelopes = None
        self.shape = None
        if os.path.isfile(self.data_name):
            self.load_data()
        else:
            self.compute_attributes()
            self.save_data()

    @abstractmethod
    def compute_attributes(self):
        pass

    @abstractmethod
    def convert_file(self):
        pass

    def save_data(self):
        data = (
            self.file_name, self.data_name,
            self.sample_rate, self.envelopes, self.envelope_times,
            self.time_series, self.sample_rate, self.shape
        )
        pickle.dump(data, open(self.data_name, 'wb'))

    def load_data(self):
        data = pickle.load(open(self.data_name, 'rb'))
        (
            self.file_name, self.data_name,
            self.sample_rate, self.envelopes, self.envelope_times,
            self.time_series, self.sample_rate, self.shape
        ) = data

    def plot(self, markers=None):
        period = 1 / self.sample_rate
        num_points = min(len(self.time_series), int(10 * self.sample_rate))
        t = np.linspace(0.0, period * num_points, num_points)
        fig = plt.figure(figsize=(40, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t, self.time_series[:num_points])
        if markers is not None:
            markers = markers[np.where(markers < num_points)]
            ax.plot(t[markers], self.time_series[:num_points][markers], 'x')
        plt.savefig(self.file_name + '.png')
        plt.show()


class Audio(Media):
    def __init__(self, file_name):
        super().__init__(file_name)
        self.extended_length = None
        self.extended_file_name = None

    def compute_attributes(self):
        self.sample_rate, data = wavfile.read(self.file_name)
        self.shape = data.shape
        assert data.dtype == dtype('int16')
        data = data / (2. ** 15)
        self.time_series = data.mean(axis=1)
        self.envelope_times, self.envelopes = get_onset_envelopes(
            self.time_series, self.sample_rate
        )

    def convert_file(self):
        if not self.file_name.endswith('.wav'):
            old_name = self.file_name
            self.file_name = self.file_name.rsplit('.', 1)[0] + '.wav'
            cmd = 'ffmpeg -i {} {}'.format(
                old_name, self.file_name
            )
            if not os.path.isfile(self.file_name):
                subprocess.call(cmd, shell=True)

    def extend(self, new_length):
        self.extended_length = new_length
        self.extended_file_name = '{}_{}.wav'.format(
            self.file_name[:-4], new_length
        )
        if not os.path.isfile(self.extended_file_name):
            sample_rate, data = wavfile.read(self.file_name)
            old_length = len(data)
            tail = new_length - old_length * (new_length // old_length)
            extended_data = np.vstack(tuple([data] * (new_length // old_length) + [data[:tail, :]]))
            wavfile.write(self.extended_file_name, sample_rate, extended_data)


class Video(Media):
    def __init__(self, file_name):
        super().__init__(file_name)

    def compute_attributes(self):
        input_stream = cv.VideoCapture(self.file_name)
        self.sample_rate = input_stream.get(cv.CAP_PROP_FPS)
        input_stream.release()
        # Use multithreading
        input_stream = FileVideoStream(self.file_name).start()
        # flows = [0.0]
        directograms = []
        directogram_times = []
        # ret, frame = input_stream.read()
        frame = input_stream.read()
        last_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.shape = frame.shape
        while input_stream.more():
            if len(directograms) % 250 == 0:
                print('Video Processing: {:.2f}s'.format(len(directograms) / self.sample_rate))
            frame = input_stream.read()
            # ret, frame = input_stream.read()
            # if not ret:
            #     break
            if not input_stream.more():
                break
            next_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(
                last_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            directograms.append(get_directogram(flow))
            directogram_times.append(len(directograms) / self.sample_rate)
            # flows.append(self._calculate_movement(flow))
            last_frame = next_frame
        self.envelopes = get_impact_envelopes(directograms)
        self.envelope_times = np.array(directogram_times)
        self.time_series = np.array([0]+list(self.envelopes))
        # input_stream.release()
        input_stream.stop()

    def convert_file(self):
        pass

    @staticmethod
    def _calculate_movement(flow):
        x_m, y_m = flow[:, :, 0], flow[:, :, 1]
        return np.linalg.norm(x_m - np.mean(x_m)) + np.linalg.norm(y_m - np.mean(y_m))
