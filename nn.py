import numpy as np
import librosa
import librosa.display
# download a sound and see what it looks like
sound, sr = librosa.load(librosa.util.example_audio_file())


#                         '1-100032-A-0.wav')

if len(sound) == 1:
    print('Mono')
else:
    print('Stereo')

## new block

import matplotlib.pyplot as plt

plt.figure(figsize = (12, 8))

D = librosa.amplitude_to_db(librosa.stft(sound), ref = np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis = 'linear')
plt.show()
