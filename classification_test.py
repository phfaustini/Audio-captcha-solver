import numpy as np
import pandas as pd
import os
import librosa
import scipy
from scipy.stats import skew
from tqdm import tqdm
tqdm.pandas()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


SAMPLE_RATE = 44100
DATA_FOLDER = 'output/'
AUDIO_FOLDERS = [DATA_FOLDER+i for i in os.listdir(DATA_FOLDER)]
AUDIO_FILENAMES = [] # -> <number>_<digit>.wav
for folder in AUDIO_FOLDERS:
    for f in os.listdir(folder):
        AUDIO_FILENAMES.append(folder+'/'+f)



# Generate mfcc features with mean and standard deviation
def get_mfcc(audio_filename: str, path: str) -> pd.core.series.Series:
    data, _ = librosa.core.load(path +'/'+ audio_filename, sr=SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    try:
        ft1 = librosa.feature.mfcc(data, sr=SAMPLE_RATE, n_mfcc=30)
        ft2 = librosa.feature.zero_crossing_rate(data)[0]
        ft3 = librosa.feature.spectral_rolloff(data)[0]
        ft4 = librosa.feature.spectral_centroid(data)[0]
        ft5 = librosa.feature.spectral_contrast(data)[0]
        ft6 = librosa.feature.spectral_bandwidth(data)[0]
        #ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.median(ft1, axis = 1), np.min(ft1, axis = 1)))
        ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
        #ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
        #ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
        #ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
        #ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
        label = audio_filename.split('.')[0].split('-')[-1]
        #return pd.Series(np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc, label)))
        #return pd.Series(np.hstack((ft2_trunc, label)))
        return pd.Series(np.hstack((ft2.size, label))) # Qtas vezes cortou o zero
    except:
        print('bad file')
        return pd.Series([0]*210)



if __name__ == "__main__":
    training_data = []
    for sample in AUDIO_FILENAMES:
        folder = '/'.join(sample.split('/')[0:2])
        filename = sample.split('/')[-1]
        obj = get_mfcc(filename, folder)
        training_data.append(obj)

    data = []
    targets = []
    for sample in training_data:
        d = sample[0:sample.size - 1]
        l = sample[sample.size - 1]
        data.append(d)
        targets.append(l)


    X_train, X_test, y_train, y_test = train_test_split(data, targets, 
                                                        test_size=0.1, 
                                                        random_state=2)

    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Acuracia Naive Bayes = "+str(sum(y_pred == y_test) / len(y_pred)))

    #for i in range(len(y_pred)):
    #    print("y_pred = "+str(y_pred[i])+" | y_test = "+str(y_test[i]))
