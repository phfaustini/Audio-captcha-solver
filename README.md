# Projeto Captcha

## trim

O arquivo trim contém os métodos responsáveis pela segmentação dos caracteres.

Foi usada a biblioteca SoX em conjunto com pysndfx (que é só uma interface bonitinha do SoX).

Para instalar o SoX, é só rodar pelo conda:

`conda install -c conda-forge sox`

Só explicando oq eu fiz:

Para detectar as regiões interessantes, apliquei alguns filtros em sequencia:

- .limiter(20.0): Qualquer coisa acima de 20dB é reduzida pra 20dB. Isso serve para reduz a diferença de volume entre os caracteres.
- .lowpass(2500, 2): Qualquer frequência acima de 2500Hz é descartada. O 2 é tipo a suavidade da curva perto de 2500.
- .highpass(100):  Qualquer frequência abaixo de 100Hz é descartada. Esses dois passos removem frequências que contém pouca informação vocal.
- .equalizer(300, db=15.0): Aqui é a parte mais importante. Na região dos 300Hz, onde a maior parte da fala ocorre, damos um boost de 15dB, o que é bastante coisa. Dessa forma, as regiões com voz ficarão bem mais altas do que o ruído (aparecem uns picos gigantes), facilitando a identificação de fronteiras.

Tem comentado alguns gráficos (a parte dos waveplot) se quiserem. A região azul é o áudio depois das transformações.

Essas transformações são só usadas para a identificação das regiões. Na hora de escrever o arquivo de saída, uso o áudio normal mesmo.

A identificação está no método agglomerative do librosa. Pego 7 slices, pq o final nunca é detectado. Logo, o oitavo fica do sétimo slice até o final.

O resto não tem muito segredo.
