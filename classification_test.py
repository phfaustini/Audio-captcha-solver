import numpy as np
import pandas as pd
import os
from math import fabs
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from spectrum import get_spectrum
from sklearn import tree # dot -Tpng tree.dot -o tree.png


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


def extract_features(audio_filename: str, path: str) -> pd.core.series.Series:
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
    
    feature5 = librosa.feature.spectral_contrast(data)
    feature5_flat = np.hstack((np.median(feature5), np.std(feature5)))

    feature6 = librosa.feature.spectral_bandwidth(data)
    feature6_flat = np.hstack((np.median(feature6), np.std(feature6)))

    feature7 = librosa.feature.tonnetz(data)
    feature7_flat = np.hstack((np.median(feature7), np.std(feature7)))


    feature8_flat = get_spectrum(data)

    #plt.figure(figsize=(10, 4))
    #librosa.display.specshow(feature1, x_axis='time')
    #plt.colorbar()
    #plt.title(label)
    #plt.tight_layout()
    #plt.show()
    
    features = pd.Series(np.hstack((feature1_flat, feature2_flat, feature3_flat, 
                                    feature4_flat, feature5_flat, feature6_flat, 
                                    feature7_flat, feature8_flat, label)))
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

    # Normalisar
    std_scale = preprocessing.StandardScaler().fit(X_train_raw) 
    X_train = std_scale.transform(X_train_raw)
    return X_train, np.array(y_train), std_scale


def test(X_train: np.ndarray, y_train: np.ndarray, std_scale: preprocessing.data.StandardScaler):
    accuracy1NN = 0
    accuracySVM = 0
    accuracyTotal = 0
    total1NN = 0
    totalSVM = 0
    total = 0
    for folder in TEST_AUDIO_CAPTCHA_FOLDERS:
        correct1NN = 0
        correctSVM = 0
        correct = 0
        for filename in os.listdir(folder):
            obj = extract_features(filename, folder)
            y_test = obj[obj.size - 1]
            X_test_raw = [obj[0:obj.size - 1]]
            X_test = std_scale.transform(X_test_raw) # Normalisar
            
            neigh1 = KNeighborsClassifier(n_neighbors=1)    #D, N
            y_pred_1nn = neigh1.fit(X_train, y_train).predict(X_test)
            #if y_pred_1nn[0] == y_test:
            #    correct1NN+=1
            #    total1NN+=1
                #print(y_test+" "+y_pred[0]) 
            #else:   
            #    print(y_test+" "+y_pred[0])

            clf = SVC()
            y_pred_svm = clf.fit(X_train, y_train).predict(X_test)              
            #if y_pred_svm[0] == y_test:
            #    correctSVM+=1
            #    totalSVM+=1
                #print(y_test+" "+y_pred[0]) 
            #else:   
                #print(y_test+" "+y_pred[0])

            y_pred = y_pred_svm[0]
            if y_pred_svm[0] == 'd' or y_pred_svm[0] == 'm': # SVM erra muito essas
                y_pred = y_pred_1nn[0]
            if y_pred_1nn[0] == 'd' or y_pred_1nn[0] == 'm':
                y_pred = y_pred_1nn[0]

            if y_pred == y_test:
                correct+=1
                total+=1
                print('V '+y_test+" "+y_pred[0])
            else:
                print('E '+y_test+" "+y_pred[0])
        #if correct1NN == 4:
        #    accuracy1NN+=1
        #if correctSVM == 4:
        #    accuracySVM+=1
        
            
        if correct == 4:
            accuracyTotal+=1

    number_of_folders = len(TEST_AUDIO_CAPTCHA_FOLDERS)
    number_of_characters = len(TEST_AUDIO_FILENAMES)
    #print("Acuracia (captcha) 1NN = {0:.2f}%".format((accuracy1NN / number_of_folders)*100))
    #print("Acuracia (captcha) SVM = {0:.2f}%".format((accuracySVM / number_of_folders)*100))
    #print("Acuracia (caracteres) 1NN = {0:.2f}%".format((total1NN / number_of_characters)*100))
    #print("Acuracia (caracteres) SVM = {0:.2f}%".format((totalSVM / number_of_characters)*100))
    print("Acuracia (captcha) = {0:.2f}%".format((accuracyTotal / number_of_folders)*100))
    print("Acuracia (caracteres) = {0:.2f}%".format((total / number_of_characters)*100))



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


def break_captcha():
    X_train, y_train, std_scale = train()
    test(X_train, y_train, std_scale)

if __name__ == "__main__":
    break_captcha()
    #a=important_features()