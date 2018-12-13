import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import spectrogram


def get_onset_envelopes(x, r):
    # http://www.abedavis.com/files/papers/VisualRhythm_Davis18.pdf
    # Section 3.1-3.2

    frequencies, times, spectro = spectrogram(x, fs=r, mode='complex')
    spectro_abs = np.absolute(spectro)
    spectro_flux = np.zeros(spectro.shape)
    spectro_flux[:, 1:] = (spectro_abs - np.roll(spectro_abs, 1, axis=1))[:, 1:]
    spectro_flux[spectro_flux < 0] = 0.0
    onset_envelope = spectro_flux.sum(axis=0)
    return times, onset_envelope


def get_directogram(f_t):
    pass


def get_impact_envelopes(directogram):
    pass


# a = Audio('input_files/audio/zeze/zeze.wav')
# start_time = time.time()
#
# a.time_series = a.time_series[:len(a.time_series)//20]
#
# times, envelope = get_onset_envelopes(
#     a.time_series,
#     a.sample_rate
# )
#
# period = 1 / a.sample_rate
# num_points = len(a.time_series)
# t = np.linspace(0.0, period * num_points, num_points)
# fig = plt.figure(figsize=(40, 5))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(t, a.time_series)
# ax.plot(times, envelope)
# plt.show()
#
#
# print(envelope)
# print('Finished in {:.2f}s'.format(time.time() - start_time))

