# -*- coding: utf-8 -*-
"""
Uso: chame a funcao get_spectrum. 

Ela retorna a intensidade média de cada um dos 7 espectros de frequência.
Será sempre um valor entre 0 e 1, normalmente mais próximo de 0 e acredito que
nunca acima de 0,5.

Fonte: https://www.teachmeaudio.com/mixing/techniques/audio-spectrum/

@author: Matheus
"""
from pysndfx import AudioEffectsChain
import numpy as np

def make_chain(low, high):
    return (AudioEffectsChain()
        .lowpass(high, 3.0)
        .highpass(low, 3.0))

sb = make_chain(20, 60)
b = make_chain(60, 250)
lm = make_chain(250, 500)
m = make_chain(500, 2000)
um = make_chain(2000, 4000)
p = make_chain(4000, 6000)
br = make_chain(6000, 20000)

specs = [sb,b,lm,m,um,p,br]


def get_spectrum(audio):
    return [np.mean(np.abs(spectrum(audio))) for spectrum in specs]
