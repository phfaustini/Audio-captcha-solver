import trim
import os

diretorio = './fase_1/base_treinamento_I/'

for f in os.listdir(diretorio):
    filename = os.path.join(diretorio, f)
    audios = trim.trim(filename, './output')