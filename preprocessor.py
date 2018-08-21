import os

import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

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


def _remove_until(l, until):
    t = list(l)
    n = len(t)
    while n > until:
        m = np.full((n, n), np.inf)

        for i in range(n):
            for j in range(i+1, n):
                m[i, j] = t[j] - t[i]

        to_adapt, to_remove = np.unravel_index(m.argmin(), m.shape)
        t[to_adapt] = (t[to_adapt]+t[to_remove])/2
        del t[to_remove]
        n = len(t)
    return t


def _extract_labels(filename):
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
    peaks = _remove_until(peaks, N_SLICES)

    if SHOW_PLOTS:
        peak_times = librosa.samples_to_time(peaks)
        plt.title(filename)
        librosa.display.waveplot(fxy, color='m')
        librosa.display.waveplot(y, color='orange')
        plt.vlines(peak_times, -1, 1, color='k', linestyle='--', linewidth=6, alpha=0.9, label='Segment boundaries')
        plt.show()
        return

    labels = _extract_labels(filename)

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
            _, [left, right] = librosa.effects.trim(audio, top_db=12, frame_length=2)
            left = int(round(max(0, left - FOLGA//4)))
            right = int(round(min(right + FOLGA//4, len(y)-1)))
            audio_trim = audio[left:right]
            audio_trim = librosa.util.normalize(audio_trim)
            librosa.output.write_wav('%s/%s/%d-%s.wav' % (output, labels, i, labels[i]), audio_trim, sr=sr)

#trim('amostras/abn7.wav', 'output')
