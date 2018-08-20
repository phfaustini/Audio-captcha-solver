import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from constants import *
from transformation import *


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
