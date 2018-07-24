import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os
from pysndfx import AudioEffectsChain
from scipy.signal import find_peaks

fx = (
    AudioEffectsChain()
    .limiter(20.0)
    .lowpass(2500, 2)
    .highpass(100)
    .equalizer(300, db=15.0)
)

#denoise = AudioEffectsChain().lowpass(3000)

def remove_until(l, until):
    t = list(l)
    n = len(t)
    while n > until:
        m = np.full((n, n), np.inf)

        for i in range(n):
            for j in range(i+1, n):
                m[i,j] = t[j] - t[i]

        to_adapt, to_remove = np.unravel_index(m.argmin(), m.shape)
        t[to_adapt] = (t[to_adapt]+t[to_remove])/2
        del t[to_remove]
        n = len(t)
    return t

def extract_labels(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def trim(filename, output, SHOW_PLOTS=False):
    y, sr = librosa.load(filename)

    FOLGA = int(sr)
    N_SLICES = 4

    y = librosa.to_mono(y)
    y = librosa.util.normalize(y)
    y, indexes = librosa.effects.trim(y, top_db=24, frame_length=2)
    fxy = fx(y)
    fxy[np.abs(fxy) < 0.5] = 0

    peaks, _ = find_peaks(fxy, height=0.5, distance=sr)
    peaks = remove_until(peaks, N_SLICES)

    if SHOW_PLOTS:
        peak_times = librosa.samples_to_time(peaks)
        plt.title(filename)
        librosa.display.waveplot(fxy)
        librosa.display.waveplot(y)
        plt.vlines(peak_times, -1, 1, color='red', linestyle='--',linewidth=8, alpha=0.9, label='Segment boundaries')
        plt.show()
        return

    labels = extract_labels(filename)

    if not os.path.exists('%s/%s' % (output, labels)):
        os.mkdir('%s/%s' % (output, labels))

    for i in range(N_SLICES):
        if(i >= len(peaks)):
            continue
        p = peaks[i]
        left = int(round(max(0, p - FOLGA)))
        right = int(round(min(p + FOLGA, len(y)-1)))
        audio = y[left:right]
        if(np.any(audio)):
            _, [l,r] = librosa.effects.trim(audio, top_db=12, frame_length=2)
            l = int(round(max(0, l - FOLGA//4)))
            r = int(round(min(r + FOLGA//4, len(y)-1)))
            audio_trim = audio[l:r]
            audio_trim = librosa.util.normalize(audio_trim)
            librosa.output.write_wav('%s/%s/%d-%s.wav' % (output, labels, i, labels[i]), audio_trim, sr=sr)

#trim('amostras/abn7.wav', 'output')
