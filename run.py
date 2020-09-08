#!/usr/bin/env python
import os

print("First of all, we need to extract features from wav-files")
print("We will use librosa - a Python lib for audio processing - for that")
import librosa

# from all the good files
for i in [j for j in os.listdir() if j.endswith('wav')]:
    print("extracting features from %s" % i)
    print("- using librosa Core API functions just to check what they provide")
    print("-- spectral representations")
    print("--- short-time fourier transform")

    break
