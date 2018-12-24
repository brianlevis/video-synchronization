import os
import pickle
import subprocess
from abc import ABC, abstractmethod

import cv2 as cv
import librosa
import matplotlib.pyplot as plt
import numpy as np
from imutils.video import FileVideoStream

from synchronize import get_directogram, get_impact_envelopes


HOP_LENGTH = 512


class Media(ABC):
    def __init__(self, file_name):
        self.file_name = file_name
        self.convert_file()
        self.data_name = self.file_name + '_data.pkl'
        self.sample_rate = None
        self.shape = None
        self.envelopes = None
        self.envelope_times = None
        self.envelope_sample_rate = None
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
            self.file_name, self.data_name, self.sample_rate, self.shape,
            self.envelopes, self.envelope_times, self.envelope_sample_rate
        )
        pickle.dump(data, open(self.data_name, 'wb'))

    def load_data(self):
        data = pickle.load(open(self.data_name, 'rb'))
        (
            self.file_name, self.data_name, self.sample_rate, self.shape,
            self.envelopes, self.envelope_times, self.envelope_sample_rate
        ) = data

    def plot(self, markers=None):
        num_points = min(len(self.envelopes), int(30 * self.sample_rate))
        fig = plt.figure(figsize=(40, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.envelope_times[:num_points], self.envelopes[:num_points])
        if markers is not None:
            markers = markers[np.where(markers < num_points)]
            ax.plot(self.envelope_times[markers], self.envelopes[:num_points][markers], 'x')
        plt.savefig(self.file_name + '.png')
        plt.show()


class Audio(Media):
    def __init__(self, file_name):
        super().__init__(file_name)
        self.extended_length = None
        self.extended_file_name = None

    def compute_attributes(self):
        time_series, self.sample_rate = librosa.load(self.file_name)
        self.shape = time_series.shape
        self.envelopes = librosa.onset.onset_strength(time_series, sr=self.sample_rate)
        self.envelope_times = librosa.frames_to_time(np.arange(len(self.envelopes)), sr=self.sample_rate)
        self.envelope_sample_rate = self.sample_rate / HOP_LENGTH

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
            data, sample_rate = librosa.load(self.file_name, mono=False)
            old_length = data.shape[1]
            tail = new_length - old_length * (new_length // old_length)
            extended_data = np.vstack(tuple([data] * (new_length // old_length) + [data[:, :tail]]))
            librosa.output.write_wav(self.extended_file_name, extended_data, sample_rate)


class Video(Media):
    def __init__(self, file_name):
        super().__init__(file_name)

    def compute_attributes(self):
        input_stream = cv.VideoCapture(self.file_name)
        self.sample_rate = input_stream.get(cv.CAP_PROP_FPS)
        input_stream.release()
        # Use multi-threading by Adrian of https://pyimagesearch.com
        input_stream = FileVideoStream(self.file_name).start()
        directograms = []
        directogram_times = []
        frame = input_stream.read()
        last_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.shape = frame.shape
        while input_stream.more():
            if len(directograms) % 250 == 0:
                print('Video Processing: {:.2f}s'.format(len(directograms) / self.sample_rate))
            frame = input_stream.read()
            if not input_stream.more():
                break
            next_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # These parameters were taken from a demo in the documentation
            flow = cv.calcOpticalFlowFarneback(
                last_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            directograms.append(get_directogram(flow))
            directogram_times.append(len(directograms) / self.sample_rate)
            last_frame = next_frame
        self.envelope_times = np.array(directogram_times)
        self.envelopes = get_impact_envelopes(directograms, self.envelope_times)
        self.envelope_sample_rate = self.sample_rate
        input_stream.stop()

    def convert_file(self):
        pass
