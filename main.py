import trim_peaks as trim
import os

TRAINING_FOLDER = './fase_1/base_treinamento_I/'
TEST_FOLDER = './fase_1/base_validacao_I/'
TRAINING_OUTPUT = './output_training/'
TEST_OUTPUT = './output_test/'

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
    from classification_test import break_captcha
    break_captcha()