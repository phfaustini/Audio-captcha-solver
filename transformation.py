from math import fabs

from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import librosa

from pysndfx import AudioEffectsChain
from constants import *


def _count_peaks(audio):
    _, perc = librosa.effects.hpss(audio)
    y = np.abs(perc)
    a2 = _moving_average(y, 441)
    #h = np.mean(a2)-np.std(a2)/2
    r0, _ = find_peaks(a2, width=441, height=0.05)
    return len(r0)


def _moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def _make_chain(low, high):
    return (AudioEffectsChain()
        .lowpass(high, 3.0)
        .highpass(low, 3.0))


sb = _make_chain(20, 60)
b = _make_chain(60, 250)
lm = _make_chain(250, 500)
m = _make_chain(500, 2000)
um = _make_chain(2000, 4000)
p = _make_chain(4000, 6000)
br = _make_chain(6000, 20000)

specs = [sb, b, lm, m, um, p, br]


def _get_spectrum(audio):
    """Uso: chame a funcao _get_spectrum.

    Ela retorna a intensidade média de cada um dos 7 espectros de frequência.
    Será sempre um valor entre 0 e 1, normalmente mais próximo de 0 e acredito que
    nunca acima de 0,5.
    """
    return [np.mean(np.abs(spectrum(audio))) for spectrum in specs]


def extract_features(audio_filename: str, path: str) -> pd.core.series.Series:
    """
    As features são retornadas concatenadas em um vetor de atributos.
    """
    data, _ = librosa.core.load(path+'/'+audio_filename, sr=SAMPLE_RATE)
    label = audio_filename.split('.')[0].split('-')[-1]

    feature1_raw = librosa.feature.mfcc(data, sr=SAMPLE_RATE, n_mfcc=40)
    feature1 = np.array([list(map(fabs, sublist)) for sublist in feature1_raw]) # all > 0
    npstd = np.std(feature1, axis=1)
    npmedian = np.median(feature1, axis=1)
    feature1_flat = np.hstack((npmedian, npstd))

    feature2 = librosa.feature.zero_crossing_rate(y=data)
    feature2_flat = feature2.size / data.size

    feature5_flat = _get_spectrum(data)

    feature6_flat = _count_peaks(data)

    sc = librosa.feature.spectral_contrast(y=data, sr=SAMPLE_RATE)

    rms = librosa.feature.rmse(data) # meio que uma media dos picos
    rms_median = np.median(rms)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    features = pd.Series(np.hstack((feature1_flat, feature2_flat,
                                    feature5_flat, feature6_flat, [np.mean(sc), np.std(sc)], 
                                    rms_median, rms_mean, rms_std, 
                                    label)))
    return features
