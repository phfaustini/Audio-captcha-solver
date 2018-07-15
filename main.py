import trim
import os
import librosa

for f in os.listdir('./amostras'):
    filename = './amostras/%s' % f
    audios = trim.trim(filename, './output')