import sys
import time

from media import Audio, Video
from peak import local_peaks, local_peaks_falling_edge
from synchronize import convert_video, synchronize_peaks


assert sys.version_info.major == 3


def synchronize_video(audio_name, video_name, output_name, replace_audio=None):
    print('{} + {} -> {}'.format(audio_name, video_name, output_name))
    start_time = time.time()
    # Load one-dimensional data and save if necessary
    audio = Audio('input_files/audio/' + audio_name)
    video = Video('input_files/video/' + video_name)
    # Compute peaks
    audio_peaks = local_peaks(audio)
    video_peaks = local_peaks_falling_edge(video)  # delay frames

    # audio.plot(markers=audio_peaks)
    # video.plot(markers=video_peaks)

    audio_peaks_sync, video_peaks_sync = synchronize_peaks(
        audio, audio_peaks, video, video_peaks
    )

    manual_audio = Audio('input_files/audio/' + replace_audio) if replace_audio is not None else None

    convert_video(audio, audio_peaks_sync, video, video_peaks_sync, 'output_files/' + output_name, manual_audio)
    print('Finished in {:.2f}s'.format(time.time() - start_time))


# synchronize_video('drum_116.wav', 'tap.mp4', 'tap_drum.mp4')
# synchronize_video('drum_116.wav', 'tap_still/tap_still.mp4', 'tap_still/tap_still_output_f.mp4')
# synchronize_video('drum_116.wav', 'tap_moving/tap_moving.mp4', 'tap_moving/tap_moving_output.mp4')
# synchronize_video('zeze/zeze.webm', 'zeze/fruit_salad.mp4', 'zeze/wiggles.mp4')
# synchronize_video('zeze/zeze_beat.webm', 'zeze/gummy.mp4', 'zeze/bear_beat_no_words.mp4')
#                   replace_audio='zeze/zeze.webm')
# synchronize_video('drum_116.wav', 'turtle/turtle.mp4', 'turtle/turtle_drum.mp4')
# synchronize_video('frank/pyramids.webm', 'frank/dance.mp4', 'frank/dancing.mp4')
synchronize_video('red/red.wav', 'car/car.mp4', 'redmercedes_audio_e.mp4')
# synchronize_video('zeze/zeze_beat.wav', 'gummy/gummy.mp4', 'gummy_zeze_beat_words.mp4', replace_audio='zeze/zeze.wav')
