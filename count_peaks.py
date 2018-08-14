from scipy.signal import find_peaks
import numpy as np

def count_peaks(audio):
    audio = np.abs(audio)
    a2 = moving_average(audio, 4410)
    #h = np.mean(a2)-np.std(a2)/2
    r0, _ = find_peaks(a2, width=441, height=0.05)
    return len(r0)

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n