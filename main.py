import trim_peaks as trim
import os

diretorio = './fase_1/base_treinamento_I/'
#diretorio = './amostras'

for f in os.listdir(diretorio):
    filename = os.path.join(diretorio, f)
    audios = trim.trim(filename, './output')