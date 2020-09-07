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
    sampling_freq, data = scipy.io.wavfile.read(wav)
    print("the wave")
    print(data)
    print("samplig frequency")
    print(sampling_freq)
    os.system('aplay %s' % wav)
    plt.specgram(data)
    plt.show()
