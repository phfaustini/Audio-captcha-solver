import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os

def remove_until(l, until):
    t = list(l)
    n = len(t)
    while n > until:
        m = np.full((n, n), np.inf)
        
        for i in range(n):
            for j in range(i+1, n):
                m[i,j] = t[j] - t[i]
        
        _, to_remove = np.unravel_index(m.argmin(), m.shape)
        del t[to_remove]
        n = len(t)
    return t

def extract_labels(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def trim(filename, output):
    y, sr = librosa.load(filename)
    
    FOLGA = sr // 2
    N_SLICES = 4
    
    y = librosa.to_mono(y)
    y = librosa.util.normalize(y)
    y, indexes = librosa.effects.trim(y, top_db=24, frame_length=2)
    S_full, _ = librosa.magphase(librosa.stft(y))
    #vocal isolation:
    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))
    
    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v = 2, 10
    power = 2
    
    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)
    
    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)
    
    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components
    
    S_foreground = mask_v * S_full
    S_background = mask_i * S_full
    
    """
    plt.figure(121)
    librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    
    plt.figure(122)
    librosa.display.specshow(librosa.amplitude_to_db(S_foreground, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    """
    bounds = librosa.segment.agglomerative(S_foreground, 7)
    bound_times = librosa.frames_to_time(bounds)
    #plt.vlines(bound_times, 0, S_foreground.shape[0], color='linen', linestyle='--',linewidth=2, alpha=0.9, label='Segment boundaries')
    
    #plt.show()
    cuts = librosa.frames_to_samples(bounds)
    labels = extract_labels(filename)
    if not os.path.exists('%s/%s' % (output, labels)):
        os.mkdir('%s/%s' % (output, labels))
    for i in range(N_SLICES):
        if i == N_SLICES-1:
            audio = y[cuts[2*i]-FOLGA:]
        elif i == 0:
            audio = y[cuts[2*i]:cuts[2*i+1]+FOLGA]
        else:
            audio = y[cuts[2*i]-FOLGA:cuts[2*i+1]+FOLGA]
        if(np.any(audio)):
            audio_trim, _ = librosa.effects.trim(audio, top_db=24, frame_length=2)
            audio_trim = librosa.util.normalize(audio_trim)
            librosa.output.write_wav('%s/%s/%s.wav' % (output, labels, labels[i]), audio_trim, sr=sr)
