import numpy as np
import scipy.signal as scisignal
from matplotlib import pyplot as plt, colors

HISTOGRAM_BINS = np.linspace(-np.pi, np.pi, 100)


def get_onset_envelopes(time_series, sample_rate, plot=False):
    # http://www.abedavis.com/files/papers/VisualRhythm_Davis18.pdf
    # Section 3.1-3.2
    if plot:
        frequencies, times, spectrogram = scisignal.spectrogram(time_series, fs=sample_rate)
        plot_colormesh(
            times, frequencies, spectrogram,
            'Time [sec]', 'Frequency [Hz]',
            file_name='spectrogram.png'
        )
    # TODO: get rid of 224x size reduction for 48kHz audio
    frequencies, times, spectrogram = scisignal.spectrogram(time_series, fs=sample_rate, mode='complex')
    spectro_abs = np.absolute(spectrogram)
    spectro_flux = np.zeros(spectrogram.shape)
    spectro_flux[:, 1:] = (spectro_abs - np.roll(spectro_abs, 1, axis=1))[:, 1:]
    spectro_flux[spectro_flux < 0] = 0.0
    if plot:
        plot_colormesh(
            times, frequencies, spectro_flux,
            'Time [sec]', 'Frequency [Hz]',
            file_name='spectro_flux.png'
        )
    onset_envelopes = spectro_flux.sum(axis=0)
    return times, onset_envelopes


def get_directogram(flow):
    norms = np.linalg.norm(flow, axis=2)
    angles = np.arctan2(flow[:, :, 1], flow[:, :, 0])
    angle_indicators = np.digitize(angles, HISTOGRAM_BINS)
    directogram = np.zeros((len(HISTOGRAM_BINS),))
    for y in range(flow.shape[0]):
        for x in range(flow.shape[1]):
            directogram[angle_indicators[y, x]] += norms[y, x]
    return directogram


def get_impact_envelopes(directograms, times, plot=False):
    directogram = np.vstack(directograms)
    filtered_directogram = scisignal.medfilt2d(directogram)
    if plot:
        plot_colormesh(
            times, HISTOGRAM_BINS, directogram.T,
            'Time [sec]', 'Angle [Rad]',
            file_name='directogram.png'
        )
    flux = np.zeros(directogram.shape)
    flux[1:, :] = (filtered_directogram - np.roll(filtered_directogram, 1, axis=0))[1:, :]
    flux[flux < 0] = 0.0
    if plot:
        plot_colormesh(
            times, HISTOGRAM_BINS, flux.T,
            'Time [sec]', 'Angle [Rad]',
            file_name='deceleration.png'
        )
    impact_envelopes = flux.sum(axis=1)
    clip_threshold = np.percentile(impact_envelopes, 98)
    impact_envelopes[impact_envelopes > clip_threshold] = 0
    impact_envelopes = impact_envelopes / impact_envelopes.max()
    return impact_envelopes


def plot_colormesh(x_axis, y_axis, z_axis, x_label, y_label, file_name=None):
    plt.clf()
    plt.pcolormesh(x_axis, y_axis, z_axis, norm=colors.PowerNorm(gamma=1. / 2.))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if file_name is not None:
        plt.savefig(file_name)
    plt.show()
