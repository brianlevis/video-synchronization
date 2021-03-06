import librosa
import numpy as np
from scipy.signal import find_peaks


LOCAL_MEAN_WINDOW = 0.2
LOCAL_MAX_WINDOW = 0.3


def librosa_peaks(media):
    hop_length = round(media.sample_rate / media.envelope_sample_rate)
    peaks = librosa.onset.onset_detect(onset_envelope=media.envelopes, sr=media.sample_rate, hop_length=hop_length)
    # Convert envelope indices into time_series indices
    peak_times = media.envelope_times[peaks]
    peak_indices = (peak_times * media.sample_rate).astype('int')
    return peak_indices


def get_beats(media):
    # T_a(n, k) = Sum_q (u_a(hn+q) u_a(hn+q+k) w(q)) / (2N+1−k)
    hop_length = round(media.sample_rate / media.envelope_sample_rate)
    # tempogram = librosa.feature.tempogram(
    #     onset_envelope=media.envelopes, sr=media.sample_rate,
    #     hop_length=hop_length,
    #     win_length=5*media.sample_rate/hop_length
    # )
    # ac_global = librosa.autocorrelate(media.envelopes, max_size=tempogram.shape[0])
    # ac_global = librosa.util.normalize(ac_global)

    # Calculate tempo as a step function of time
    tempo = librosa.beat.tempo(
        onset_envelope=media.envelopes,
        sr=media.sample_rate,
        hop_length=hop_length,
        ac_size=5.0,
        aggregate=None
    )
    assert len(tempo) == len(media.envelopes)
    beat_stack = []
    section_begin = 0
    for i in range(len(media.envelopes)):
        section_tempo = tempo[section_begin]
        if i == len(tempo) - 1 or tempo[i] != section_tempo:
            if i - section_begin >= 5:
                # Find beats for each tempo section
                _, section_beat_indices = librosa.beat.beat_track(
                    onset_envelope=media.envelopes[section_begin:i],
                    sr=media.sample_rate,
                    hop_length=hop_length,
                    trim=False,
                    bpm=section_tempo
                )
                # The beats correspond to envelope indices for some reason
                section_start_time = media.envelope_times[section_begin]
                beat_times = section_start_time + section_beat_indices / media.envelope_sample_rate
                global_beat_indices = (beat_times * media.sample_rate).astype(np.int)
                beat_stack.append(global_beat_indices)
            section_begin = i
    beats = np.hstack(beat_stack)
    # peak_times = media.envelope_times[beats]
    # peak_indices = (peak_times * media.sample_rate).astype('int')
    return beats


def impacts(media):
    """Find impacts as specified in Visual Rhythm and Beats."""
    global_max = media.envelopes.max()
    mean_window_delta = int(LOCAL_MEAN_WINDOW / 2 * media.sample_rate)
    max_window_delta = int(LOCAL_MAX_WINDOW / 2 * media.sample_rate)
    peaks = []
    for i in range(max_window_delta + 4, len(media.envelope_times) - max_window_delta - 4):
        local_mean = (
            media.envelopes[i - mean_window_delta:i].mean() +
            media.envelopes[i+1:i+1 + mean_window_delta].mean()
        ) / 2
        local_max = max(
            media.envelopes[i - max_window_delta:i].max(),
            media.envelopes[i+1:i+1 + max_window_delta].max()
        )
        current = media.envelopes[i]
        if current > local_max and (current - local_mean) > 0.1 * global_max:
            peaks.append(i)
    return np.array(peaks)


def local_envelope_peaks(media):
    peaks, _ = find_peaks(
        media.envelopes, distance=int(media.envelope_sample_rate / 4)
    )
    # Convert envelope indices into time_series indices
    peak_times = media.envelope_times[peaks]
    peak_indices = (peak_times * media.sample_rate).astype('int')
    return peak_indices

#
# def local_peaks(media):
#     time_series = media.envelopes
#     sample_rate = media.sample_rate
#     # peaks, _ = find_peaks(time_series, distance=min_distance * sample_rate)
#     # kernel = cv2.getGaussianKernel(int(sample_rate / 8), int(sample_rate / 8))
#     # time_series = np.convolve(time_series, kernel.flatten())
#     peaks, _ = find_peaks(
#         time_series, distance=int(sample_rate / 4)  # , height=time_series.mean()
#     )
#     return peaks
#
#
# def local_peaks_falling_edge(media):
#     time_series = media.envelopes
#     sample_rate = media.sample_rate
#     # peaks, _ = find_peaks(time_series, distance=min_distance * sample_rate)
#     # kernel = cv2.getGaussianKernel(int(sample_rate / 8), int(sample_rate / 8))
#     # time_series = np.convolve(time_series, kernel.flatten())
#     # media.time_series = time_series
#     peaks, _ = find_peaks(
#         time_series, distance=int(sample_rate / 4)  # , height=time_series.mean()
#     )
#     falling_peaks = np.zeros_like(peaks)
#     for p_i in range(len(peaks)):
#         peak = peaks[p_i] + 2
#         while peak + 1 < len(time_series) and time_series[peak] > time_series[peak + 1]:
#             peak += 1
#         falling_peaks[p_i] = peak
#     return falling_peaks
#
#
# def z_score_peak(y, lag=7, threshold=5, influence=0.3):
#     # Source: https://stackoverflow.com/a/22640362/6029703
#     num_points = len(y)
#     signals = np.zeros((num_points,))
#     filtered = np.zeros((num_points,))
#     average = np.zeros((num_points,))
#     std = np.zeros((num_points,))
#     average[lag - 1] = np.mean(y[0:lag])
#     std[lag - 1] = np.std(y[0:lag])
#     for i in range(lag, len(y)):
#         if abs(y[i] - average[i-1]) > threshold * std[i-1]:
#             if y[i] > average[i-1]:
#                 signals[i] = 1
#             filtered[i] = influence * y[i] + (1 - influence) * filtered[i-1]
#             average[i] = np.mean(filtered[(i-lag+1):i+1])
#             std[i] = np.std(filtered[(i-lag+1):i+1])
#         else:
#             signals[i] = 0
#             filtered[i] = y[i]
#             average[i] = np.mean(filtered[(i-lag+1):i+1])
#             std[i] = np.std(filtered[(i-lag+1):i+1])
#     return signals
