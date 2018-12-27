import sys
import time

from media import Audio, Video
from peak import local_envelope_peaks, librosa_peaks, get_beats
from synchronize import combine_videos, convert_video, synchronize_peaks

assert sys.version_info.major == 3


def synchronize_video(audio_name, video_name, output_name, replace_audio=None, plot=False):
    print('{} + {} -> {}'.format(audio_name, video_name, output_name))
    start_time = time.time()
    # Load one-dimensional data and save if necessary
    audio = Audio('input_files/audio/' + audio_name)
    video = Video('input_files/video/' + video_name)
    # Compute peaks
    audio_peaks = librosa_peaks(audio)
    video_peaks = librosa_peaks(video)

    audio_beats = get_beats(audio)
    video_beats = get_beats(video)

    if plot:
        audio.plot(markers=audio_peaks)
        video.plot(markers=video_peaks)
        audio.plot(markers=audio_beats)
        video.plot(markers=video_beats)

    audio_peaks_sync, video_peaks_sync = synchronize_peaks(
        audio, audio_beats, video, video_peaks
    )

    manual_audio = Audio('input_files/audio/' + replace_audio) if replace_audio is not None else None

    convert_video(audio, audio_peaks_sync, video, video_peaks_sync, 'output_files/' + output_name, manual_audio)
    print('Finished in {:.2f}s'.format(time.time() - start_time))


synchronize_video('red/red.wav', 'car/car.mp4', 'smol_beat_test3.mp4')

# combine_videos(
#     'output_files/best/redmercedes_beat.mp4',
#     'input_files/video/car/car.mp4',
#     'output_files/best/car_s-b-s.mp4',
#     'input_files/audio/red/beat.webm'
# )
