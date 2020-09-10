import librosa, librosa.display
import math
import matplotlib.pyplot as plt

SAMPLE_RATE = 22050

def load_sound(filename):
    # load the sound at the default sample rate 22050 HZ
    sound, sr = librosa.load(
        filename
    )
    assert sr == SAMPLE_RATE
    return sound, sr

# Calculate the MFCCs over the segments, a.k.a. frames. Prepare the
# parameters for calculating the MFCCs over the segments. In the video,
# 10 frames per 30 sec was used, I have 5 sec, but let me use 5 frames.
NUM_FRAMES = 10
# 5 / 5 = 1 seconds per frame;
# 22050 samples per frame.
frame_length_in_samples = int(SAMPLE_RATE / NUM_FRAMES)
print(frame_length_in_samples)

def extract_mfccs_from_track(sound, sr):
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

            # these are somewhat magic constants; IDK what they mean. It seems
            # redundant to me.
            n_fft = 2048,
            hop_length = 512,
        )

        num_mfcc_vectors_per_segment = math.ceil(
            frame_length_in_samples / 512
        )

        print("{}vs{}".format(len(mfcc), num_mfcc_vectors_per_segment))
        # should always be the same, but in the video we
        # check if the length is not equal to expected length.

        # librosa.display.specshow(mfcc)
        # plt.show()

def prepare_data(root):
    data = {
        # a range of mfccs that represent the sound
        "mfcc": [],
        # labels
        "label": [],
        # filenames
        "name": []
    }
