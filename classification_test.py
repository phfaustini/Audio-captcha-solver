import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.stats import skew
from tqdm import tqdm
tqdm.pandas()
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

SAMPLE_RATE = 44100
TRAINING_OUTPUT = 'output_training/'
TRAINING_AUDIO_CAPTCHA_FOLDERS = [TRAINING_OUTPUT+i for i in os.listdir(TRAINING_OUTPUT)]
TRAINING_AUDIO_FILENAMES = [] # -> <number>_<digit>.wav
for folder in TRAINING_AUDIO_CAPTCHA_FOLDERS:
    for f in os.listdir(folder):
        TRAINING_AUDIO_FILENAMES.append(folder+'/'+f)

TEST_OUTPUT = 'output_test/'
TEST_AUDIO_CAPTCHA_FOLDERS = [TEST_OUTPUT+i for i in os.listdir(TEST_OUTPUT)]


def extract_features(audio_filename: str, path: str) -> pd.core.series.Series:
    data, _ = librosa.core.load(path +'/'+ audio_filename, sr=SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    try:
        label = audio_filename.split('.')[0].split('-')[-1]

        ft1 = librosa.feature.mfcc(data, sr=SAMPLE_RATE, n_mfcc=40)
        npstd = np.std(ft1, axis=1)
        npmedian = np.median(ft1, axis=1)
        delta = np.max(ft1) - np.min(ft1)
        ft1_trunc = np.hstack((npmedian, delta))

        ft2 = librosa.feature.zero_crossing_rate(y=data)
        ft2_trunc = ft2.size
        
        #plt.figure(figsize=(10, 4))
        #librosa.display.specshow(ft1, x_axis='time')
        #plt.colorbar()
        #plt.title(label)
        #plt.tight_layout()
        #plt.show()
        
        features = pd.Series(np.hstack((ft1_trunc, ft2_trunc, label)))
        return features
    except:
        print('bad file')
        return pd.Series([0]*210)


def train() -> tuple:
    X_train_raw = []
    y_train = []
    for sample in TRAINING_AUDIO_FILENAMES:
        folder = '/'.join(sample.split('/')[0:2])
        filename = sample.split('/')[-1]
        obj = extract_features(filename, folder)
        d = obj[0:obj.size - 1]
        l = obj[obj.size - 1]
        X_train_raw.append(d)
        y_train.append(l)

    # Normalise
    std_scale = preprocessing.StandardScaler().fit(X_train_raw) 
    X_train = std_scale.transform(X_train_raw)
    return X_train, y_train, std_scale


def test(X_train: np.ndarray, y_train: np.ndarray, std_scale: preprocessing.data.StandardScaler):
    accuracyNB = 0
    accuracy1NN = 0
    accuracy3NN = 0
    accuracySVM = 0
    for folder in TEST_AUDIO_CAPTCHA_FOLDERS:
        correctNB = 0
        correct1NN = 0
        correct3NN = 0
        correctSVM = 0
        for filename in os.listdir(folder):
            obj = extract_features(filename, folder)
            y_test = obj[obj.size - 1]
            X_test_raw = [obj[0:obj.size - 1]]
            X_test = std_scale.transform(X_test_raw) # normalise
            
            gnb = GaussianNB()
            y_pred = gnb.fit(X_train, y_train).predict(X_test)
            if y_pred[0] == y_test:
                correctNB+=1
            
            neigh1 = KNeighborsClassifier(n_neighbors=1)
            y_pred = neigh1.fit(X_train, y_train).predict(X_test)
            if y_pred[0] == y_test:
                correct1NN+=1

            neigh3 = KNeighborsClassifier(n_neighbors=1)
            y_pred = neigh3.fit(X_train, y_train).predict(X_test)
            if y_pred[0] == y_test:
                correct1NN+=1

            clf = SVC()
            y_pred = clf.fit(X_train, y_train).predict(X_test)
            if y_pred[0] == y_test:
                correctSVM+=1
        
        if correctNB == 4:
            accuracyNB+=1
        if correct1NN == 4:
            accuracy1NN+=1
        if correct3NN == 4:
            accuracy3NN+=1
        if correctSVM == 4:
            accuracySVM+=1

    print("Acuracia 1NN = "+str(accuracy1NN / len(TEST_AUDIO_CAPTCHA_FOLDERS)))
    print("Acuracia 1NN = "+str(accuracy3NN / len(TEST_AUDIO_CAPTCHA_FOLDERS)))
    print("Acuracia SVM = "+str(accuracySVM / len(TEST_AUDIO_CAPTCHA_FOLDERS)))
    print("Acuracia Naive Bayes = "+str(accuracyNB / len(TEST_AUDIO_CAPTCHA_FOLDERS)))



def important_features() -> np.ndarray:
    """Retorna um array com as features mais importantes,
    extraidas a partir da base de treino.
    """
    X, Y, std_scale = train()
    rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    rnd_clf.fit(X, Y)
    print(rnd_clf.feature_importances_)
    return rnd_clf.feature_importances_


def break_captcha():
    X_train, y_train, std_scale = train()
    test(X_train, y_train, std_scale)

if __name__ == "__main__":
    break_captcha()
    #a=important_features()
