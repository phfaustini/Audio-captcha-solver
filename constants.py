import os

TRAINING_FOLDER = 'fase_II/base_treinamento_II/'
VALIDATION_FOLDER = 'fase_II/base_validacao_II/'
TEST_FOLDER = 'fase_II/base_teste_II/'
TRAINING_OUTPUT = 'output_training/'
VALIDATION_OUTPUT = 'output_validation/'
TEST_OUTPUT = 'output_test/'


if not os.path.exists(TRAINING_OUTPUT):
    os.mkdir(TRAINING_OUTPUT)
if not os.path.exists(VALIDATION_OUTPUT):
    os.mkdir(VALIDATION_OUTPUT)
if not os.path.exists(TEST_OUTPUT):
    os.mkdir(TEST_OUTPUT)

SAMPLE_RATE = 44100


TRAINING_AUDIO_CAPTCHA_FOLDERS = [TRAINING_OUTPUT+i for i in os.listdir(TRAINING_OUTPUT)]
TRAINING_AUDIO_FILENAMES = []  # -> <number>_<digit>.wav
for folder in TRAINING_AUDIO_CAPTCHA_FOLDERS:
    for f in os.listdir(folder):
        TRAINING_AUDIO_FILENAMES.append(folder+'/'+f)


VALIDATION_AUDIO_CAPTCHA_FOLDERS = [VALIDATION_OUTPUT+i for i in os.listdir(VALIDATION_OUTPUT)]
VALIDATION_AUDIO_FILENAMES = []  # -> <number>_<digit>.wav
for folder in VALIDATION_AUDIO_CAPTCHA_FOLDERS:
    for f in os.listdir(folder):
        VALIDATION_AUDIO_FILENAMES.append(folder+'/'+f)


TEST_AUDIO_CAPTCHA_FOLDERS = [TEST_OUTPUT+i for i in os.listdir(TEST_OUTPUT)]
TEST_AUDIO_FILENAMES = []  # -> <number>_<digit>.wav
for folder in TEST_AUDIO_CAPTCHA_FOLDERS:
    for f in os.listdir(folder):
        TEST_AUDIO_FILENAMES.append(folder+'/'+f)