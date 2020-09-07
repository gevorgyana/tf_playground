import numpy as np
import librosa
# download a sound and see what it looks like
sound = librosa.load('1-100032-A-0.wav')

if len(sound[0].shape) == 1:
    print('Mono')
else:
    print('Stereo')
