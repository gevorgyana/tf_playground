import librosa

wave, sr = librosa.load(
    '1-110389-A-0.wav',
    sr = None
)

print(sr)
