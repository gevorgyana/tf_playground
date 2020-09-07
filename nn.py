import numpy as np
import librosa
# download a sound and see what it looks like
sound = librosa.load('1-100032-A-0.wav')

if len(sound[0].shape) == 1:
    print('Mono')
else:
    print('Stereo')

# experiment with all the different ways to visualize sound with plots
# in frequency/time domain, + spectrograms

# use spectrogram to do ml on it
