import sys
import time

from media import Audio, Video
from peak import local_envelope_peaks, impacts
from synchronize import convert_video, synchronize_peaks


assert sys.version_info.major == 3


def synchronize_video(audio_name, video_name, output_name, replace_audio=None):
    print('{} + {} -> {}'.format(audio_name, video_name, output_name))
    start_time = time.time()
    # Load one-dimensional datra and save if necessary
    audio = Audio('input_files/audio/' + audio_name)
    video = Video('input_files/video/' + video_name)
    # Compute peaks
    audio_peaks = local_envelope_peaks(audio)
    video_peaks = local_envelope_peaks(video)

    audio.plot(markers=audio_peaks)
    video.plot(markers=video_peaks)

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
# synchronize_video('zeze/zeze_beat.webm', 'gummy/gummy.mp4', 'zeze/bear_beat_en.mp4',
#                   replace_audio='zeze/zeze.webm')
# synchronize_video('drum_116.wav', 'turtle/turtle.mp4', 'turtle/turtle_drum.mp4')
# synchronize_video('frank/pyramids.webm', 'frank/dance.mp4', 'frank/dancing.mp4')
# synchronize_video('red/red.wav', 'car/car.mp4', 'redmercedes_impacts.mp4')
# synchronize_video('zeze/zeze.wav', 'gummy/gummy.mp4', 'zeze_gummy_impact.mp4')
# synchronize_video('zeze/zeze_beat.wav', 'gummy/gummy.mp4', 'zeze_gummy_impact_beat.mp4', replace_audio='zeze/zeze.wav')

synchronize_video('beat/beat.m4a', 'army/army.mp4', 'favorites/beatit.mp4')
synchronize_video('clean_mix/short/cello.mp3', 'geese/geese.mp4', 'favorites/honk_honk.mp4')
synchronize_video('clean_mix/01_ric_short.m4a', 'gummy/happy_smol.mp4', 'favorites/ric_flair.mp4')
synchronize_video('club/club.m4a', 'gummy/gummy.mp4', 'favorites/kids.mp4')
synchronize_video('clean_mix/clean_mix_2.wav', 'ballet/ballet.mp4', 'favorites/ballet.mp4')

