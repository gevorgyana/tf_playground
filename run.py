#!/usr/bin/env python
import os

print("First of all, we need to extract features from wav-files\n"
      "We will use librosa - a Python lib for audio processing")

import librosa

print("For plotting, we need - matplotlib\n - a wrapper from librosa")
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

DISPLAY_PLOT_WAVE = False
DISPLAY_PLOT_STFT = False
DISPLAY_WITH_MANUALLY_CHANGED_SCALE = False

# should always be True for the scirpt to work
POWER_TO_DB = True

def demo_librosa():
    # for all the good files
    for i in [j for j in os.listdir() if j.endswith('wav')]:
        print("loading the wav using default sampling rate = 22050 %s"
              % i)

        wav_i, sr = librosa.load(i)
        assert sr == 22050

        print("- using librosa Core API functions to check what they provide")
        print("-- spectral representations")
        print("--- short-time fourier transform a.k.a. `librosa.stft`\n"
              "--- using the deafult sample rate length of the windowed\n"
              "--- signal (TODO change later) and other defaults")

        if DISPLAY_PLOT_WAVE == True:
            librosa.display.waveplot(wav_i)
            plt.show()

        # hop length = 512 is a smart default that gives the most precision
        # https://dsp.stackexchange.com/questions/248/how-do-i-optimize-the-window-lengths-in-stft, also may need the default value too
        stft = librosa.stft(wav_i, hop_length = 512)

        if DISPLAY_PLOT_STFT == True:
            librosa.display.specshow(stft)
            plt.show()

        print("- What happens if we take the absolute values?")
        stft = np.abs(stft)
        if DISPLAY_PLOT_STFT == True:
            librosa.display.specshow(stft)
            plt.title("Now showing the absolute values")
            plt.show()
        print("- stft is now np.abs(stft)")

        print("- Manually change the y-scale to Decebel scale")
        stft_log = librosa.amplitude_to_db(stft, ref = np.max)

        if DISPLAY_WITH_MANUALLY_CHANGED_SCALE == True:
            librosa.display.specshow(stft_log, sr=sr,
                                     x_axis = 'time', y_axis = 'log')
            plt.show()

        # Seems like the previous image is an MFCC, and there are many
        # ways to create one in librosa, but I will use the funcition that
        # is specifically tailored for that

        print("Calculate the MFCC")
        # - As suggested by docs, TODO tweak `fmax` - it may be necessary to
        # remove all of the zero-ed out freqeuncies at the top of MFCC.
        # - Use this method, not custom-made conversions!!!!!

        S = librosa.feature.melspectrogram(wav_i, sr)
        # The example in the docs runs this too, but IDK if that is needed.

        # Disapling this will lead to a bad output - so we need this thing
        # to get a prper MFCC
        if POWER_TO_DB == True:
            S = librosa.power_to_db(S, ref = np.max)
            librosa.display.specshow(S, x_axis = 'time', y_axis = 'mel',
                                     sr = sr, fmax = 8000)
            plt.show()
        print("Successful")

# demo_librosa()

'''
print("\n --- \n")
print("Preparing the dataset : building a native Python dictionary")

def prepare_data():
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(".")):
        print("{} {} {} {}".format(i, dirpath, dirnames, filenames))

prepare_data()
'''
