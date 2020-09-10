#!/usr/bin/env python

import librosa, librosa.display
import math
import os
import matplotlib.pyplot as plt

SAMPLE_RATE = 22050

def load_sound(filename):
    # load the sound at the default sample rate 22050 HZ
    sound, sr = librosa.load(
        filename
    )
    assert sr == SAMPLE_RATE
    return sound, sr

# EXPERIMENT LATER: For now, we have 1 sec of sample, but it makes no
# sense to do this, because the sounds usually take up to 3 seconds
# approximately. In the reference project on speech recognition, they were
# using the dataset that had 1 sec words in each track.
# TODO: Completely remove segmenting!!! It is useless here.

NUM_FRAMES = 2
frame_length_in_samples = int(SAMPLE_RATE / NUM_FRAMES)
print(frame_length_in_samples)

def extract_mfccs_from_track(sound, sr):
    mfccs = []

    # calculate the MFCCs over the frames
    for i in range(NUM_FRAMES):
        start_sample = i * frame_length_in_samples
        end_sample = start_sample + frame_length_in_samples

        print("{}:{}".format(start_sample, end_sample))

        frame = sound[start_sample:end_sample]

        mfcc = librosa.feature.mfcc(

            frame,
            sr,

            # may be increased to get more granular information, but 13 is
            # the minimum value
            n_mfcc = 13,

            # these are somewhat magic constants; IDK what they mean.
            # It seems redundant to me. TODO: check the Slack channel,
            # in case they anser.
            n_fft = 2048,
            hop_length = 512,
        )

        assert 13 == len(mfcc)
        print("hm frames in this sample {}",
              frame_length_in_samples / 512
        )

        num_mfcc_vectors_per_segment = math.ceil(
            frame_length_in_samples / 512
        )

        # librosa.display.specshow(mfcc)
        # plt.show()

        mfcc = mfcc.T

        # should always be the same, but in the video we
        # check if the length is not equal to expected length.
        # this is the number of frames that we obtained in one segment
        assert len(mfcc) == num_mfcc_vectors_per_segment

        print("Have {} frames inside of this segment, all of them must\n"
              "be seen on the plot", len(mfcc))

        mfccs.append(mfcc)

    return mfccs

DATA_NEEDED_CNT = 10

def prepare_data(root, path_to_csv):

    data = {
        # a range of mfccs that represent the sound
        "mfcc": [],
        # labels
        "label": [],
        # filenames
        "name": []
    }

    entries_counter = 0

    # map filename to mfccs, and label
    for i, (path, dirnames, filenames) in enumerate(os.walk(root)):
        if not path.endswith('audio'):
            continue
        for f in filenames:
            if not f.endswith('wav'):
                continue
            print("file name: {}", f.split(".")[0])
            # find the label of the record
            with open(path_to_csv, 'r') as meta_data:
                for s in meta_data:
                    splits = s.split(',')
                    if splits[0] == f:
                        label = splits[3]
                        break
            print(label)

            print(f"{path}/{f}")

            # calculate mfccs for this file
            sound, sr = load_sound(f"{path}/{f}")
            mfccs = extract_mfccs_from_track(sound, sr)

            print("- hm segments we have in this mfcc {}", len(mfccs))

            data["name"].append(f)
            data["label"].append(label)
            data["mfcc"].append(mfccs)

            entries_counter += 1

            if entries_counter == DATA_NEEDED_CNT:
                break

    return data

data = prepare_data('.', 'kaggle_ds/esc50.csv')

print(data)
