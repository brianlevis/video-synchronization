import math
import os
import subprocess

import cv2 as cv
import numpy as np


SPEEDUP_RANGE = (0.3, 3)


def convert_video(audio, audio_peaks, video, video_peaks, output_filename, manual_audio):
    # Extend audio to appropriate length
    audio.extend(audio_peaks[-1] + 1)
    # Initialize video streams
    if os.path.isfile(output_filename):
        os.remove(output_filename)
    temp_file_name = output_filename + '_temp.mp4'
    input_stream = cv.VideoCapture(video.file_name)
    output_stream = cv.VideoWriter(
        temp_file_name, cv.VideoWriter_fourcc(*'mp4v'),
        video.sample_rate, (video.shape[1], video.shape[0])
    )
    # Iterate through each output frame until the final peak is reached
    last_audio_peak, last_video_peak = 0, 0
    input_frame_index = 0
    ret, input_frame = input_stream.read()
    assert ret
    for audio_peak, video_peak in zip(audio_peaks, video_peaks):
        # Output time range
        audio_start_time = last_audio_peak / audio.sample_rate
        audio_end_time = audio_peak / audio.sample_rate
        # Input time range
        video_start_time = last_video_peak / video.sample_rate
        video_end_time = video_peak / video.sample_rate
        # Conversion multiplier
        multiplier = (
            video_end_time - video_start_time
        ) / (
            audio_end_time - audio_start_time
        )
        # Iterate through every output frame in the current peak range
        output_start_index = math.ceil(audio_start_time * video.sample_rate)
        output_end_index = math.floor(audio_end_time * video.sample_rate)
        for output_index in range(output_start_index, output_end_index + 1):
            output_time = output_index / video.sample_rate
            input_time = (output_time - audio_start_time) * multiplier + video_start_time
            input_index = round(input_time * video.sample_rate)
            while input_frame_index < input_index:
                ret, input_frame = input_stream.read()
                input_frame_index += 1
                assert ret
            output_stream.write(input_frame)
        last_audio_peak, last_video_peak = audio_peak, video_peak
    # Close video streams
    input_stream.release()
    output_stream.release()
    # Combine video and audio
    if manual_audio is None:
        audio_file = audio.extended_file_name
    else:
        manual_audio.extend(audio_peaks[-1] + 1)
        audio_file = manual_audio.extended_file_name
    cmd = 'ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental {}'.format(
        temp_file_name, audio_file, output_filename
    )
    subprocess.call(cmd, shell=True)
    os.remove(temp_file_name)


def synchronize_peaks(audio, audio_peaks, video, video_peaks, plot=False):
    audio_intervals = peaks_to_intervals(audio_peaks, audio)
    video_intervals = peaks_to_intervals(video_peaks, video)
    assert np.array_equal(audio_peaks, intervals_to_peaks(audio_intervals))
    assert np.array_equal(video_peaks, intervals_to_peaks(video_intervals))
    audio_interval_len = len(audio_intervals)
    audio_intervals_sync = []
    video_intervals_sync = []
    # Loop through each peak-separated section
    audio_index = 0
    video_index = 0
    audio_interval = audio_intervals[audio_index]
    video_interval = video_intervals[video_index]
    while True:
        video_time = video_interval / video.sample_rate
        audio_time = audio_interval / audio.sample_rate
        speed_change = video_time / audio_time
        if speed_change < SPEEDUP_RANGE[0]:
            # If the video is too short
            video_index += 1
            if video_index == len(video_intervals):
                break
            video_interval = video_intervals[video_index]
        elif SPEEDUP_RANGE[1] < speed_change:
            # If the audio is too short
            audio_index += 1
            audio_interval += audio_intervals[audio_index % audio_interval_len]
        else:
            audio_intervals_sync.append(audio_interval)
            video_intervals_sync.append(video_interval)
            audio_index += 1
            video_index += 1
            if video_index == len(video_intervals):
                break
            audio_interval = audio_intervals[audio_index % audio_interval_len]
            video_interval = video_intervals[video_index]

    audio_peaks_sync = intervals_to_peaks(audio_intervals_sync)
    video_peaks_sync = intervals_to_peaks(video_intervals_sync)
    if plot:
        video.plot(markers=video_peaks_sync)
    return audio_peaks_sync, video_peaks_sync


def peaks_to_intervals(peaks, media):
    first = [peaks[0]]
    middle = [peaks[i+1] - peaks[i] for i in range(len(peaks) - 1)]
    last = [len(media.time_series) - peaks[-1]]
    return first + middle + last


def intervals_to_peaks(intervals):
    peaks = [intervals[0]]
    for i in intervals[1:-1]:
        peaks.append(peaks[-1] + i)
    return peaks


def combine_videos(left_video_name, right_video_name, output_name, audio_name):
    cmd = """
        ffmpeg \
        -i {} \
        -i {} \
        -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
        -map [vid] \
        -c:v libx264 \
        -crf 23 \
        -preset veryfast \
        {}_temp.mp4
        """.format(
        left_video_name, right_video_name, output_name
    )
    subprocess.call(cmd, shell=True)
    cmd = """
            ffmpeg -i {}_temp.mp4 \
            -i {} \
            -c:v copy -c:a aac -strict experimental \
            {}
        """.format(
        output_name, audio_name, output_name
    )
    subprocess.call(cmd, shell=True)
    subprocess.call("rm {}_temp.mp4".format(output_name), shell=True)


