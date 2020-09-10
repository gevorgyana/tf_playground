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

NUM_FRAMES = 1
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

        # tolist() so that we can store this in JSON
        mfccs.append(mfcc.tolist())

    return mfccs

DATA_NEEDED_CNT = 3000

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

import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

# store the data in a file - UNCOMMENT WHEN RUNNING FOR THE FIRST TIME

'''
data = prepare_data('.', 'kaggle_ds/esc50.csv')
print("data prepared")
with open('data.json', 'w') as out:
    json.dump(data, out)
'''

def plot_training_results(history):
    fig, axs = plt.subplots(2)
    axs[0].plot(history.history['accuracy'], label = 'train accuracy')
    axs[0].plot(history.history['val_accuracy'], label = 'test accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc = 'lower right')
    axs[0].set_title('Accuracy eval')

    axs[1].plot(history.history['loss'], label = 'train error')
    axs[1].plot(history.history['val_loss'], label = 'test error')
    axs[1].set_ylabel('Error')
    axs[1].legend(loc = 'lower right')
    axs[1].set_title('Error eval')

    plt.show()

# now read the data
with open('data.json', 'r') as data:
    data_json = json.load(data)
    def fill_one_hot(answer_index, total_classes):
        one_hot = []
        for i in range(total_classes):
            if answer_index == i:
                one_hot.append(1)
            else:
                one_hot.append(0)
        return one_hot
    current_code = 0
    vis = []
    label2code = {}
    for k in data_json['label']:
        # print(k)
        if k in vis:
            continue
        else:
            label2code[k] = current_code
            current_code += 1
            vis.append(k)
    one_hots = []
    for i in range(len(label2code.keys())):
        # print(i)
        for j in label2code.keys():
            # print("{} {}".format(j, label2code[j]))
            if label2code[j] == i:
                one_hots.append(
                    fill_one_hot(i, len(label2code.keys()))
                )
    X = np.array(
        data_json['mfcc']
    )
    y = np.array(data_json['label'])
    # convert to one-hots
    y = np.array(list(map(
        lambda x: one_hots[label2code[x]],
        y
    )))

    # TODO: refactor - remove the excessive dimension that comes from
    # segmenting the tracks - there is no need to split a 5second sound!
    # For now just reshape this array, but fix later.
    X = np.reshape(X,
        (X.shape[0],
         X.shape[2],
         X.shape[3]
        )
    )

    # Split the data into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.3)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    model = keras.Sequential([

        # input with LSTM
        keras.layers.LSTM(
            64,
            input_shape = (X.shape[1], X.shape[2]),
            return_sequences = True
        ),

        # 1 more LSTM, as in the video
        keras.layers.LSTM(64),

        # dense layer
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dropout(0.3),

        # output
        keras.layers.Dense(len(vis),
                           activation = 'softmax'
        )
    ])

    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)

    model.compile(optimizer,
                  # ATTENTION: Not sparse! In the tutorial, they use
                  # sparse, because they don't do one-hot encoding!
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy']
    )

    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)

    history = model.fit(
        X_train, y_train, validation_data = (X_test, y_test),
        epochs = 50,

        # !!!!!!! BATHCH SIZE IS IMPORTANT
        # previously, the value of 32 causes an error, because batch was
        # greater than 10 - the dimensionality (first index) of my toy
        # dataset. OTOH, in the tutorial, they did not have any probelm
        # because their dataset was already huge.
        batch_size = 1
    )

    plot_training_results(history)
