import os

TRAINING_FOLDER = 'fase_II/base_treinamento_II/'
TEST_FOLDER = 'fase_II/base_validacao_II/'
TEST_OUTPUT = 'output_test/'

SAMPLE_RATE = 44100
TRAINING_OUTPUT = 'output_training/'
TRAINING_AUDIO_CAPTCHA_FOLDERS = [TRAINING_OUTPUT+i for i in os.listdir(TRAINING_OUTPUT)]
TRAINING_AUDIO_FILENAMES = []  # -> <number>_<digit>.wav
for folder in TRAINING_AUDIO_CAPTCHA_FOLDERS:
    for f in os.listdir(folder):
        TRAINING_AUDIO_FILENAMES.append(folder+'/'+f)

TEST_OUTPUT = 'output_test/'
TEST_AUDIO_CAPTCHA_FOLDERS = [TEST_OUTPUT+i for i in os.listdir(TEST_OUTPUT)]

TEST_AUDIO_FILENAMES = []  # -> <number>_<digit>.wav
for folder in TEST_AUDIO_CAPTCHA_FOLDERS:
    for f in os.listdir(folder):
        TEST_AUDIO_FILENAMES.append(folder+'/'+f)
