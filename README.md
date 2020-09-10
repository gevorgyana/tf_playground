Heavily inspired by this incredible YouTube channel.
https://www.youtube.com/channel/UCZPFjMe1uRSirmSpznqvJfQ

Done:
- show spectrograms
- prepare the whole dataset into a stadard form, insipired by this video:
https://www.youtube.com/watch?v=szyGiObZymo
- simple network that successfully compiles

Todo:
- Fight overfitting
- Use LSTMs

--- [quote]

The dataset consists of 50 WAV files sampled at 16KHz for 50 different
classes.

To each one of the classes, corresponds 40 audio sample of 5 seconds each.
All of these audio files have been concatenated by class in order to have
50 wave files of 3 min. 20sec.

---

[Download this](https://www.kaggle.com/mmoreaux/environmental-sound-classification-50)
And place in the same directory as where this projects is located.


~~run `chmod +x ./preps.sh && ./preps.sh` to start playing with the examples.~~