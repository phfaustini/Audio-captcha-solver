import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier

from constants import *
from transformation import *

number_of_folders = len(TEST_AUDIO_CAPTCHA_FOLDERS)
number_of_characters = len(TEST_AUDIO_FILENAMES)


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


def _create_classifier() -> VotingClassifier:
        clf1 = KNeighborsClassifier(n_neighbors=1)
        clf2 = SVC(random_state=100)
        clf3 = LinearDiscriminantAnalysis()
        clf4 = KNeighborsClassifier(n_neighbors=3)

        #creates ensemble
        return VotingClassifier(estimators=[('1nn', clf1), ('svm', clf2), ('lda', clf3), ('3nn', clf4)], 
                                weights=[1, 1, 2, 1], 
                                n_jobs=-1) # Parallel in each core


def test(X_train: np.ndarray, y_train: np.ndarray, std_scale: preprocessing.data.StandardScaler) -> tuple:
    wrong        = {'6': 0, '7': 0, 'a': 0, 'b': 0, 'c': 0, 'd': 0, 'h': 0, 'm': 0, 'n': 0, 'x': 0}
    correct      = {'6': 0, '7': 0, 'a': 0, 'b': 0, 'c': 0, 'd': 0, 'h': 0, 'm': 0, 'n': 0, 'x': 0}
    elements     = {'6': 0, '7': 0, 'a': 0, 'b': 0, 'c': 0, 'd': 0, 'h': 0, 'm': 0, 'n': 0, 'x': 0}

    captchas_correct = 0
    characters_correct = 0

    eclf = _create_classifier()
    eclf = eclf.fit(X_train, y_train)

    for folder in TEST_AUDIO_CAPTCHA_FOLDERS:
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
