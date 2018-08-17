import numpy as np
from pysndfx import AudioEffectsChain
import pandas as pd
import os
from math import fabs
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier

from count_peaks import count_peaks

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

SAMPLE_RATE = 44100
TRAINING_OUTPUT = 'output_training/'
TRAINING_AUDIO_CAPTCHA_FOLDERS = [TRAINING_OUTPUT+i for i in os.listdir(TRAINING_OUTPUT)]
TRAINING_AUDIO_FILENAMES = [] # -> <number>_<digit>.wav
for folder in TRAINING_AUDIO_CAPTCHA_FOLDERS:
    for f in os.listdir(folder):
        TRAINING_AUDIO_FILENAMES.append(folder+'/'+f)

TEST_OUTPUT = 'output_test/'
TEST_AUDIO_CAPTCHA_FOLDERS = [TEST_OUTPUT+i for i in os.listdir(TEST_OUTPUT)]

TEST_AUDIO_FILENAMES = [] # -> <number>_<digit>.wav
for folder in TEST_AUDIO_CAPTCHA_FOLDERS:
    for f in os.listdir(folder):
        TEST_AUDIO_FILENAMES.append(folder+'/'+f)

number_of_folders = len(TEST_AUDIO_CAPTCHA_FOLDERS)
number_of_characters = len(TEST_AUDIO_FILENAMES)


def make_chain(low, high):
    return (AudioEffectsChain()
        .lowpass(high, 3.0)
        .highpass(low, 3.0))

sb = make_chain(20, 60)
b = make_chain(60, 250)
lm = make_chain(250, 500)
m = make_chain(500, 2000)
um = make_chain(2000, 4000)
p = make_chain(4000, 6000)
br = make_chain(6000, 20000)

specs = [sb,b,lm,m,um,p,br]
def get_spectrum(audio):
    """Uso: chame a funcao get_spectrum.

    Ela retorna a intensidade média de cada um dos 7 espectros de frequência.
    Será sempre um valor entre 0 e 1, normalmente mais próximo de 0 e acredito que
    nunca acima de 0,5.
    """
    return [np.mean(np.abs(spectrum(audio))) for spectrum in specs]

def extract_features(audio_filename: str, path: str) -> pd.core.series.Series:
    """
    Extrai features dos áudios a partir da biblioteca librosa.
    As features são então concatenadas em um vetor de atributos.
    """

    data, _ = librosa.core.load(path +'/'+ audio_filename, sr=SAMPLE_RATE)

    label = audio_filename.split('.')[0].split('-')[-1]

    feature1_raw = librosa.feature.mfcc(data, sr=SAMPLE_RATE, n_mfcc=40)

    feature1 = np.array([list(map(fabs, sublist)) for sublist in feature1_raw]) # Tudo positivo

    npstd = np.std(feature1, axis=1)
    npmedian = np.median(feature1, axis=1)
    feature1_flat = np.hstack((npmedian, npstd))

    feature2 = librosa.feature.zero_crossing_rate(y=data)
    feature2_flat = feature2.size / data.size

    feature5_flat = get_spectrum(data)

    feature6_flat = count_peaks(data)

    features = pd.Series(np.hstack((feature1_flat, feature2_flat,
                                    feature5_flat, feature6_flat, label)))
    return features

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
    std_scale = preprocessing.StandardScaler().fit(X_train_raw)
    X_train = std_scale.transform(X_train_raw)
    return X_train, np.array(y_train), std_scale


def create_classifier():
        clf1 = KNeighborsClassifier(n_neighbors=1)
        clf2 = RandomForestClassifier(random_state=100)
        clf3 = SVC(random_state=100)
        clf4 = LinearDiscriminantAnalysis()
        clf5 = KNeighborsClassifier(n_neighbors=3)

        #creates ensemble
        return VotingClassifier(estimators=[('1nn', clf1), ('rf', clf2), ('svm', clf3), ('lda', clf4), ('5nn', clf5)], voting='hard')

def test(X_train: np.ndarray, y_train: np.ndarray, std_scale: preprocessing.data.StandardScaler) -> tuple:
    errados      = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}
    corretos     = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}
    elementos    = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}

    accuracyTotal = 0
    total = 0

    eclf = create_classifier()
    eclf = eclf.fit(X_train, y_train)

    for folder in TEST_AUDIO_CAPTCHA_FOLDERS:
        correct = 0
        for filename in os.listdir(folder):
            obj = extract_features(filename, folder)
            y_test = obj[obj.size - 1]
            X_test_raw = [obj[0:obj.size - 1]]
            X_test = std_scale.transform(X_test_raw)
            elementos[y_test] += 1
            y_pred = eclf.predict(X_test)
            if y_pred[0] == y_test:
                correct+=1
                total+=1
                corretos[y_test] += 1
            else:
                errados[y_test] += 1

        if correct == 4:
            accuracyTotal+=1

    captchas_total = (accuracyTotal / number_of_folders)*100
    acuracia_caracteres = (total / number_of_characters)*100

    return (captchas_total, acuracia_caracteres,
            errados,corretos,
            elementos)
