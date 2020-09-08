#!/usr/bin/env python
import os

print("First of all, we need to extract features from wav-files\n"
      "We will use librosa - a Python lib for audio processing")

import librosa

print("For plotting, we need - matplotlib\n - a wrapper from librosa")
import matplotlib.pyplot as plt
import librosa.display

DISPLAY_PLOT_WAVE = True
DISPLAY_PLOT_STFT = True

# for all the good files
for i in [j for j in os.listdir() if j.endswith('wav')]:
    print("loading the wav using default sampling rate = 22050 %s" % i)

    wav_i, sr = librosa.load(i)
    assert sr == 22050

    print("- using librosa Core API functions just to check what they provide")
    print("-- spectral representations")
    print("--- short-time fourier transform a.k.a. `librosa.stft`\n"
          "--- using the deafult sample rate length of the windowed\n"
          "--- signal (TODO change later) and other defaults")

    if DISPLAY_PLOT_WAVE == True:
        librosa.display.waveplot(wav_i)
        plt.show()

    stft = librosa.stft(wav_i)

    if DISPLAY_PLOT_STFT == True:
        librosa.display.specshow(stft)
        plt.show()

    print("Todo use MFCC, but spectrograms are enough")

    break
