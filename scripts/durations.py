import wave
import contextlib
import os

for dirpath, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('wav'):
            with contextlib.closing(wave.open(f, 'r')) as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration = frames / float(rate)
                print(duration)
