Explaining the MFCC construction process.

Suppose we have a range of samples, usually we have the sample rate of
22050 Hertz. It means that we have 22050 samples per second.

What we can do with this is use the Fourier transform over the subranges
of the whole waveform. By doing so we can visualize the frequency
components that constitute the waveform, over periods of time.

We know that each track in the dataset takes 30 seconds. It means that we
have 22050 * 30 samples per a track. We need to decide how we split these
22050 * 30 samples into subranges, uniformly. Therefore we choose the
number of segments, and divide 22050 * 30 by it, thus obtaining the length
of each segment. Equivalently, we could choose the length of a segment,
and try to divide the range by it. The numbers may not be integers, but
suppose they are, so that we divide the range evenly into an integer
number of contiguous windows.

Now we want to calculate MFCC over each of the windows. Again, for that,
we have to specify yet another size of the window for the FFT, which works
with the whole waveform in batches, showing the static overall picture
depicting the frequency-magnitude information accumulated over that range.

We choose the window size of 2048. But that is not enough! MFCC has to
accept a parameter called hop_length, which is
