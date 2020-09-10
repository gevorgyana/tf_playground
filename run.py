#!/usr/bin/env python
import os

print("First of all, we need to extract features from wav-files\n"
      "We will use librosa - a Python lib for audio processing")

import librosa

print("For plotting, we need - matplotlib\n - a wrapper from librosa")
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

print
(
    "The initial dataset contains .wavs, we need Mel Spectrograms\n"
    "Therefore we need to build them first.\n"
    "But we need to get markers from the .csv file that comes with the "
    "dataset"
)

SAMPLE_RATE = 16000 # hm samples per second
DURATION = 5 # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Returns the dictionary mapping each number into its corresponding MFCCs
def prepare_mfccs():
    id2mfcc = {}

    num_samples_per_segment = int(
        SAMPLES_PER_TRACK / num_segments
    )

    expected_num_mfcc_vectors_per_segment = math.ceil(
        num_samples_per_segment / 512 # 512 is the hop_length
        # there are hop_length units inside the segment, which are
        # used for calculating MFCC
    )

    for dirpath, dirnames, filenames in os.walk("kaggle_ds"):
        if dirpath == '.':
            continue

        counter = 0
        bound = 3

        for f in filenames:
            if counter == bound:
                break
            if f.endswith('wav'):
                counter += 1
                waveform, sr = librosa.load("./{}/{}".format(dirpath, f))
                ''' this is the old way of doing it - calculating Mel
                    spectrogram instead of using MFCCs.
                id2mfcc[f] = librosa.power_to_db (
                    librosa.feature.melspectrogram(waveform, sr = sr),
                    ref = np.max
                )
                '''
                # See the docs for this function, it is customizable, but
                # I am using defaults. There is a difference between
                # mfcc and melspectrogram, but he chooses to use
                # the former ¯\_(ツ)_/¯
                # [#] https://www.youtube.com/watch?v=szyGiObZymo
                # id2mfcc[f] = librosa.feature.mfcc(waveform, sr = sr)
                for s in range(num_segments):
                    start_sample = s * num_samples_per_segment
                    finish_sample = start_sample + num_samples_per_segment
                    mfcc = librosa.feature.mfcc(
                        signal[start_sample:finish_sample],
                        sr = sr

                        # customize the form of MFCCs:
                        # n_fft = 2048,
                        # n_mfcc = 13, #
                        # hop_length = 512,

                    )

                    # TODO in the video, MFCCs are transposed

    return id2mfcc

def prepare_data():

    prepared_data = {
        "mfccs": prepare_mfccs(),
        "names": [],
        "labels": [],
    }

    for i, (dirpath, dirnames, filenames) in enumerate (
            os.walk("kaggle_ds")
    ):
        if dirpath == '.':
            continue

        for f in filenames:
            if f.endswith('wav'):
                prepared_data["names"].append (
                    f
                )

    prepared_data["names"] = prepared_data["names"][:3]

    print("alive!")

    # fill in the labels from the .csv file inside of `kaggle_ds`
    for dirpath, _, files in os.walk("kaggle_ds"):
        for f in files:
            if not f.endswith('csv'):
                continue

            csv_name = "./{}/{}".format(dirpath, f)
            print(csv_name)
            with open(csv_name) as csv:
                for i in csv:
                    columns = i.split(',')
                    name = columns[0]

                    # NOTODO quadratic algorithm is okay, Python is slow
                    # anyway :S

                    for name_picked in prepared_data['names']:
                        if name_picked == name:
                            # need its label only? TODO think about it
                            label = columns[3]

                            prepared_data['labels'].append(
                                label
                            )

    print(prepared_data)
    return prepared_data

prepare_data()
