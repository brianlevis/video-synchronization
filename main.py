import sys
import time

from media import Audio, Video
from peak import local_envelope_peaks, impacts
from synchronize import combine_videos, convert_video, synchronize_peaks


assert sys.version_info.major == 3


def synchronize_video(audio_name, video_name, output_name, replace_audio=None):
    print('{} + {} -> {}'.format(audio_name, video_name, output_name))
    start_time = time.time()
    # Load one-dimensional data and save if necessary
    audio = Audio('input_files/audio/' + audio_name)
    video = Video('input_files/video/' + video_name)
    # Compute peaks
    audio_peaks = local_envelope_peaks(audio)
    video_peaks = local_envelope_peaks(video)

    # audio.plot(markers=audio_peaks)
    # video.plot(markers=video_peaks)

    audio_peaks_sync, video_peaks_sync = synchronize_peaks(
        audio, audio_peaks, video, video_peaks
    )

    manual_audio = Audio('input_files/audio/' + replace_audio) if replace_audio is not None else None

    convert_video(audio, audio_peaks_sync, video, video_peaks_sync, 'output_files/' + output_name, manual_audio)
    print('Finished in {:.2f}s'.format(time.time() - start_time))


# synchronize_video('beat/beat.m4a', 'army/army.mp4', 'favorites/beatit.mp4')
# synchronize_video('clean_mix/short/cello.mp3', 'geese/geese.mp4', 'favorites/honk_honk.mp4')
# synchronize_video('clean_mix/01_ric_short.m4a', 'gummy/happy_smol.mp4', 'favorites/ric_flair.mp4')
# synchronize_video('club/club.m4a', 'gummy/gummy.mp4', 'favorites/kids.mp4')
# synchronize_video('clean_mix/clean_mix_2.wav', 'ballet/ballet.mp4', 'favorites/ballet.mp4')
synchronize_video('red/beat.webm', 'car/car_smol.mp4', 'smol_beat_test.mp4')
# combine_videos(
#     'output_files/best/redmercedes_beat.mp4',
#     'input_files/video/car/car.mp4',
#     'output_files/best/car_s-b-s.mp4',
#     'input_files/audio/red/beat.webm'
# )
