import os

import preprocessing as trim
from constants import *


def create_folder_structure():
    """Cria a estrutura de pastas
    necessaria, caso ela nao exista.
    Em seguida, faz a segmentacao.

    As pastas VALIDATION_OUTPUT, TEST_OUTPUT
    e TRAINING_OUTPUT contem os captchas 
    segmentados.
    """
    if not os.path.exists(TRAINING_OUTPUT):
        os.mkdir(TRAINING_OUTPUT)
    if not os.path.exists(VALIDATION_OUTPUT):
        os.mkdir(VALIDATION_OUTPUT)
    if not os.path.exists(TEST_OUTPUT):
        os.mkdir(TEST_OUTPUT)

    for f in os.listdir(TRAINING_FOLDER):
        filename = os.path.join(TRAINING_FOLDER, f)
        audios = trim.trim(filename, TRAINING_OUTPUT)
    for f in os.listdir(VALIDATION_FOLDER):
        filename = os.path.join(VALIDATION_FOLDER, f)
        audios = trim.trim(filename, VALIDATION_OUTPUT)
    for f in os.listdir(TEST_FOLDER):
        filename = os.path.join(TEST_FOLDER, f)
        audios = trim.trim(filename, TEST_OUTPUT)


if __name__ == "__main__":
    #create_folder_structure()
    from model import train, test, get_final_model, final_test
    #X_train, y_train, std_scale = train()
    #captchas_total, acuracia_caracteres, errados, corretos, elementos = test(X_train, y_train, std_scale)
    #print("Accuracia captchas: {}".format(captchas_total))
    #print("Accuracia caracteres: {}".format(acuracia_caracteres))
    final_classifier, std_scale = get_final_model()
    final_test(std_scale, final_classifier)