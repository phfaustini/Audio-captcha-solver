import os

import preprocessing as trim
from constants import *


def create_folder_structure():
    """Cria a estrutura de pastas
    necessaria, caso ela nao exista.
    Em seguida, faz a segmentacao.

    As pastas TEST_OUTPUT e TRAINING_OUTPUT
    contem os captchas segmentados.
    """
    if not os.path.exists(TRAINING_OUTPUT):
        os.mkdir(TRAINING_OUTPUT)
    if not os.path.exists(TEST_OUTPUT):
        os.mkdir(TEST_OUTPUT)

    for f in os.listdir(TRAINING_FOLDER):
        filename = os.path.join(TRAINING_FOLDER, f)
        audios = trim.trim(filename, TRAINING_OUTPUT)
    for f in os.listdir(TEST_FOLDER):
        filename = os.path.join(TEST_FOLDER, f)
        audios = trim.trim(filename, TEST_OUTPUT)


if __name__ == "__main__":
    create_folder_structure()
    from model import train, test
    X_train, y_train, std_scale = train()
    captchas_total, acuracia_caracteres, errados, corretos, elementos = test(X_train, y_train, std_scale)
    print("Accuracia captchas: {}".format(captchas_total))
    print("Accuracia caracteres: {}".format(acuracia_caracteres))
