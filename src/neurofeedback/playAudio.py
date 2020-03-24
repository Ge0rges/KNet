from pydub import AudioSegment
from pydub.playback import play
# import alsaaudio
import time

audio_data = AudioSegment.from_mp3("parade ft. yama.mp3")

# for i in range(100):
#     seg = audio_data[i:i + 100] - (60 - (60 * (i/100.0)))
#     play_obj = _play_with_simpleaudio(seg)
#     play_obj.wait_done()

play_obj = play(audio_data)

m = alsaaudio.Mixer()
vol = m.getvolume()

print(vol)
for i in range(1, 100):
    m.setvolume(i)
    time.sleep(0.1)

play_obj.wait_done()
