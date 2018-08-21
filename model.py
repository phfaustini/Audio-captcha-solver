import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

from transformation import *
from constants import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

TRAINING_AUDIO_CAPTCHA_FOLDERS = [TRAINING_OUTPUT+i for i in os.listdir(TRAINING_OUTPUT)]
def init_training_audio_filenames():
    lst = []
    for folder in TRAINING_AUDIO_CAPTCHA_FOLDERS:
        for f in os.listdir(folder):
            lst.append(folder+'/'+f)
    return lst
TRAINING_AUDIO_FILENAMES = init_training_audio_filenames()  # -> <number>_<digit>.wav


VALIDATION_AUDIO_CAPTCHA_FOLDERS = [VALIDATION_OUTPUT+i for i in os.listdir(VALIDATION_OUTPUT)]
def init_validation_audio_filenames():
    lst = []
    for folder in VALIDATION_AUDIO_CAPTCHA_FOLDERS:
        for f in os.listdir(folder):
            lst.append(folder+'/'+f)
    return lst
VALIDATION_AUDIO_FILENAMES = init_validation_audio_filenames()  # -> <number>_<digit>.wav


TEST_AUDIO_CAPTCHA_FOLDERS = [TEST_OUTPUT+i for i in os.listdir(TEST_OUTPUT)]
def init_test_audio_filenames():
    lst = []
    for folder in TEST_AUDIO_CAPTCHA_FOLDERS:
        for f in os.listdir(folder):
            lst.append(folder+'/'+f)
    return lst
TEST_AUDIO_FILENAMES = init_test_audio_filenames()  # -> <number>_<digit>.wav


def train() -> tuple:
    X_train_raw = []
    y_train = []
    for sample in TRAINING_AUDIO_FILENAMES:
        folder = '/'.join(sample.split('/')[0:2])
        filename = sample.split('/')[-1]
        obj = extract_features(filename, folder)
        features = obj[0:obj.size - 1]
        label = obj[obj.size - 1]
        X_train_raw.append(features)
        y_train.append(label)
    std_scale = preprocessing.StandardScaler().fit(X_train_raw)
    X_train = std_scale.transform(X_train_raw)
    return X_train, np.array(y_train), std_scale


def individual_classifiers_results(X_train: np.ndarray, y_train: np.ndarray, std_scale: preprocessing.data.StandardScaler) -> tuple:
    elementos    = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}

    corretos_svm = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}
    corretos_1nn = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}
    errados_svm  = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}
    errados_1nn  = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}

    corretos_3nn = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}
    corretos_lda = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}
    errados_3nn  = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}
    errados_lda  = {'6':0, '7':0, 'a':0, 'b':0, 'c':0, 'd':0, 'h':0, 'm':0, 'n':0, 'x':0}

    number_of_folders = len(VALIDATION_AUDIO_CAPTCHA_FOLDERS)
    number_of_characters = len(VALIDATION_AUDIO_FILENAMES)

    accuracy1NN = 0
    accuracySVM = 0
    total1NN = 0
    totalSVM = 0

    accuracy3NN = 0
    accuracyLDA = 0
    total3NN = 0
    totalLDA = 0

    k1nn = KNeighborsClassifier(n_neighbors=1)
    k1nn = k1nn.fit(X_train, y_train)
    svm = SVC()
    svm = svm.fit(X_train, y_train)
    lda = LinearDiscriminantAnalysis()
    lda = lda.fit(X_train, y_train)
    k3nn = KNeighborsClassifier(n_neighbors=3)
    k3nn = k3nn.fit(X_train, y_train)

    for folder in VALIDATION_AUDIO_CAPTCHA_FOLDERS:
        correctSVM = 0
        correctLDA = 0
        correct3NN = 0
        correct1NN = 0
        for filename in os.listdir(folder):
            obj = extract_features(filename, folder)
            y_test = obj[obj.size - 1]
            X_test_raw = [obj[0:obj.size - 1]]
            X_test = std_scale.transform(X_test_raw)
            elementos[y_test] += 1


            y_pred_1nn = k1nn.predict(X_test)
            if y_pred_1nn[0] == y_test:
                correct1NN+=1
                total1NN+=1
                corretos_1nn[y_test] += 1
            else:
                errados_1nn[y_test] += 1

            y_pred_svm = svm.predict(X_test)
            if y_pred_svm[0] == y_test:
                correctSVM+=1
                totalSVM+=1
                corretos_svm[y_test] += 1
            else:
                errados_svm[y_test] += 1

            y_pred_3nn = k3nn.predict(X_test)
            if y_pred_3nn[0] == y_test:
                correct3NN+=1
                total3NN+=1
                corretos_3nn[y_test] += 1
            else:
                errados_3nn[y_test] += 1

            y_pred_lda = lda.predict(X_test)
            if y_pred_lda[0] == y_test:
                correctLDA+=1
                totalLDA+=1
                corretos_lda[y_test] += 1
            else:
                errados_lda[y_test] += 1

        if correct1NN == 4:
            accuracy1NN+=1
        if correctSVM == 4:
            accuracySVM+=1
        if correct3NN == 4:
            accuracy3NN+=1
        if correctLDA == 4:
            accuracyLDA+=1

    captchas_svm = (accuracySVM / number_of_folders)*100
    caracteres_svm = (totalSVM / number_of_characters)*100
    captchas_1nn = (accuracy1NN / number_of_folders)*100
    caracteres_1nn = (total1NN / number_of_characters)*100
    captchas_3nn = (accuracy3NN / number_of_folders)*100
    caracteres_3nn = (total3NN / number_of_characters)*100
    captchas_lda = (accuracyLDA / number_of_folders)*100
    caracteres_lda = (totalLDA / number_of_characters)*100

    return (captchas_svm, caracteres_svm,
            captchas_1nn, caracteres_1nn,
            captchas_3nn, caracteres_3nn,
            captchas_lda, caracteres_lda,
            elementos,
            corretos_svm,corretos_1nn,
            errados_svm,errados_1nn,
            corretos_lda,corretos_3nn,
            errados_lda,errados_3nn)


def _create_classifier(n_jobs=-1) -> VotingClassifier:
        clf1 = KNeighborsClassifier(n_neighbors=1)
        clf2 = SVC(random_state=100)
        clf3 = LinearDiscriminantAnalysis()
        clf4 = KNeighborsClassifier(n_neighbors=3)

        #creates ensemble
        return VotingClassifier(estimators=[('1nn', clf1), ('svm', clf2), ('lda', clf3), ('3nn', clf4)], 
                                weights=[1, 1, 2, 1], 
                                n_jobs=n_jobs)  # Parallel in each core if n_jobs = -1


def test(X_train: np.ndarray, y_train: np.ndarray, std_scale: preprocessing.data.StandardScaler) -> tuple:
    wrong        = {'6': 0, '7': 0, 'a': 0, 'b': 0, 'c': 0, 'd': 0, 'h': 0, 'm': 0, 'n': 0, 'x': 0}
    correct      = {'6': 0, '7': 0, 'a': 0, 'b': 0, 'c': 0, 'd': 0, 'h': 0, 'm': 0, 'n': 0, 'x': 0}
    elements     = {'6': 0, '7': 0, 'a': 0, 'b': 0, 'c': 0, 'd': 0, 'h': 0, 'm': 0, 'n': 0, 'x': 0}

    number_of_folders = len(VALIDATION_AUDIO_CAPTCHA_FOLDERS)
    number_of_characters = len(VALIDATION_AUDIO_FILENAMES)

    captchas_correct = 0
    characters_correct = 0

    eclf = _create_classifier()
    eclf = eclf.fit(X_train, y_train)

    for folder in VALIDATION_AUDIO_CAPTCHA_FOLDERS:
        correct_preditions = 0
        for filename in os.listdir(folder):
            obj = extract_features(filename, folder)
            y_test = obj[obj.size - 1]
            X_test_raw = [obj[0:obj.size - 1]]
            X_test = std_scale.transform(X_test_raw)
            elements[y_test] += 1
            y_pred = eclf.predict(X_test)
            if y_pred[0] == y_test:
                correct_preditions += 1
                characters_correct += 1
                correct[y_test] += 1
            else:
                wrong[y_test] += 1

        if correct_preditions == 4:
            captchas_correct += 1

    accuracy_captcha = (captchas_correct / number_of_folders)*100
    accuracy_character = (characters_correct / number_of_characters)*100

    return (accuracy_captcha, accuracy_character, wrong, correct, elements)


############################################################################


def get_final_model():
    """
    Usa toda a base (treino + validação) disponível
    para treinar o modelo. Este é o modelo a ser 
    usado na base de teste final.

    Como a base disponível é toda usada, é retornada 
    também um std_scale para normalização.

    :retorno: (eclf, std_scale)
    """
    X_raw = []
    y = []
    for sample in TRAINING_AUDIO_FILENAMES:
        folder = '/'.join(sample.split('/')[0:2])
        filename = sample.split('/')[-1]
        obj = extract_features(filename, folder)
        features = obj[0:obj.size - 1]
        label = obj[obj.size - 1]
        X_raw.append(features)
        y.append(label)
    for sample in VALIDATION_AUDIO_FILENAMES:
        folder = '/'.join(sample.split('/')[0:2])
        filename = sample.split('/')[-1]
        obj = extract_features(filename, folder)
        features = obj[0:obj.size - 1]
        label = obj[obj.size - 1]
        X_raw.append(features)
        y.append(label)
    std_scale = preprocessing.StandardScaler().fit(X_raw)
    X = std_scale.transform(X_raw)

    eclf = _create_classifier(n_jobs=1)
    # DESCOMENTAR PARA TESTE DE VALIDACAO CRUZADA (NIVEL DE CARACTERE)
    #scores = cross_val_score(eclf, X, np.array(y), cv=10, n_jobs=-1)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #print("")
    #print(scores)

    eclf = eclf.fit(X, np.array(y))
    return eclf, std_scale


def final_test(std_scale, model):
    wrong        = {'6': 0, '7': 0, 'a': 0, 'b': 0, 'c': 0, 'd': 0, 'h': 0, 'm': 0, 'n': 0, 'x': 0}
    correct      = {'6': 0, '7': 0, 'a': 0, 'b': 0, 'c': 0, 'd': 0, 'h': 0, 'm': 0, 'n': 0, 'x': 0}
    elements     = {'6': 0, '7': 0, 'a': 0, 'b': 0, 'c': 0, 'd': 0, 'h': 0, 'm': 0, 'n': 0, 'x': 0}

    number_of_folders = len(TEST_AUDIO_CAPTCHA_FOLDERS)
    number_of_characters = len(TEST_AUDIO_FILENAMES)

    captchas_correct = 0
    characters_correct = 0

    for folder in TEST_AUDIO_CAPTCHA_FOLDERS:
        correct_preditions = 0
        for filename in os.listdir(folder):
            obj = extract_features(filename, folder)
            y_test = obj[obj.size - 1]
            X_test_raw = [obj[0:obj.size - 1]]
            X_test = std_scale.transform(X_test_raw)
            elements[y_test] += 1
            y_pred = model.predict(X_test)
            if y_pred[0] == y_test:
                correct_preditions += 1
                characters_correct += 1
                correct[y_test] += 1
            else:
                wrong[y_test] += 1

        if correct_preditions == 4:
            captchas_correct += 1

    accuracy_captcha = (captchas_correct / number_of_folders)*100
    accuracy_character = (characters_correct / number_of_characters)*100

    return (accuracy_captcha, accuracy_character, wrong, correct, elements)
