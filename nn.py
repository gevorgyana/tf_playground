import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

# get the WAVs
import os
from os.path import isfile, join
wavs = [wav for wav in os.listdir('.')
        if isfile(join('./', wav)) & join('./', wav).endswith('wav')]
print(wavs)

for wav in wavs:
    sampling_freq, data_orig = scipy.io.wavfile.read(wav)

    os.system('aplay %s' % wav)
    plt.specgram(data_orig)
    plt.show()

# todo mfc transform + conv net in Keras
