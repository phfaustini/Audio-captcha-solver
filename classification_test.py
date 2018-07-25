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



def extract_features_librosa(audio_filename: str, path: str) -> pd.core.series.Series:
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
    feature2_flat = feature2.size

    feature3 = librosa.feature.spectral_rolloff(data)
    feature3_flat = np.hstack((np.median(feature3), np.std(feature3)))

    feature4 = librosa.feature.spectral_centroid(data)
    feature4_flat = np.hstack((np.median(feature4), np.std(feature4)))

    feature5_flat = get_spectrum(data)
    
    features = pd.Series(np.hstack((feature1_flat, feature2_flat, feature3_flat, 
                                    feature4_flat, feature5_flat, label)))
    return features
 


def train() -> tuple:
    X_train_raw = []
    y_train = []
    for sample in TRAINING_AUDIO_FILENAMES:
        folder = '/'.join(sample.split('/')[0:2])
        filename = sample.split('/')[-1]
        obj = extract_features_librosa(filename, folder)
        d = obj[0:obj.size - 1]
        l = obj[obj.size - 1]
        X_train_raw.append(d)
        y_train.append(l)
    std_scale = preprocessing.StandardScaler().fit(X_train_raw) 
    X_train = std_scale.transform(X_train_raw)
    return X_train, np.array(y_train), std_scale


def test(X_train: np.ndarray, y_train: np.ndarray, std_scale: preprocessing.data.StandardScaler) -> tuple:
    errados      = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}
    corretos     = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}
    elementos    = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}
    corretos_svm = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}
    corretos_1nn = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}
    errados_svm  = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}
    errados_1nn  = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}

    accuracyTotal = 0
    total = 0
    accuracy1NN = 0
    accuracySVM = 0
    total1NN = 0
    totalSVM = 0
    for folder in TEST_AUDIO_CAPTCHA_FOLDERS:
        correct = 0
        correct1NN = 0
        correctSVM = 0
        for filename in os.listdir(folder):
            obj = extract_features_librosa(filename, folder)
            y_test = obj[obj.size - 1]
            X_test_raw = [obj[0:obj.size - 1]]
            X_test = std_scale.transform(X_test_raw)
            elementos[y_test] += 1                
                
            neigh1 = KNeighborsClassifier(n_neighbors=1)    #D, N
            y_pred_1nn = neigh1.fit(X_train, y_train).predict(X_test)
            if y_pred_1nn[0] == y_test:
                correct1NN+=1
                total1NN+=1
                corretos_1nn[y_test] += 1
            else:
                errados_1nn[y_test] += 1

            clf = SVC()
            y_pred_svm = clf.fit(X_train, y_train).predict(X_test)
            if y_pred_svm[0] == y_test:
                correctSVM+=1
                totalSVM+=1
                corretos_svm[y_test] += 1
            else:
                errados_svm[y_test] += 1       

            y_pred = y_pred_svm[0]
            if y_pred_svm[0] == 'd' or y_pred_svm[0] == 'm': # SVM erra muito essas
                y_pred = y_pred_1nn[0]
            if y_pred_1nn[0] == 'd' or y_pred_1nn[0] == 'm':
                y_pred = y_pred_1nn[0]

            if y_pred == y_test:
                correct+=1
                total+=1
                corretos[y_test] += 1
            else:
                errados[y_test] += 1
            
        if correct1NN == 4:
            accuracy1NN+=1
        if correctSVM == 4:
            accuracySVM+=1
        if correct == 4:
            accuracyTotal+=1

    captchas_total = (accuracyTotal / number_of_folders)*100
    acuracia_caracteres = (total / number_of_characters)*100
    captchas_svm = (accuracySVM / number_of_folders)*100
    caracteres_svm = (totalSVM / number_of_characters)*100
    captchas_1nn = (accuracy1NN / number_of_folders)*100
    caracteres_1nn = (total1NN / number_of_characters)*100

    return (captchas_total, acuracia_caracteres, 
            captchas_svm, caracteres_svm, 
            captchas_1nn, caracteres_1nn, 
            errados,corretos,
            elementos,
            corretos_svm,corretos_1nn,
            errados_svm,errados_1nn)


def important_features() -> np.ndarray:
    """Retorna um array com as features mais importantes,
    extraidas a partir da base de treino.
    """
    X, Y, std_scale = train()
    rnd_clf = RandomForestClassifier(n_estimators=1000, max_features=1, n_jobs=-1, random_state=42)
    rnd_clf.fit(X, Y)
    importances = rnd_clf.feature_importances_
    print(importances)
    return importances
