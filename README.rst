****************
Projeto Captcha
****************

Estrutura de pastas necessária:
-------------------------------
* **fase_II/** - Pasta original com os arquivos de áudio.
* **fase_II/base_treinamento_II** - Diretório com dados de treinamento.
* **fase_II/base_validacao_II** - Diretório com dados de validação.
* **fase_II/base_teste_II** - Diretório com dados de teste.

Execução:
---------
* Digitar::

    jupyter notebook

Isso abrirá o relatório. Ao final dele, há uma linha::

    final_classifier, std_scale = get_final_model()

Ela carrega o modelo **final** a ser usado contra a base de teste. Para testá-lo, basta executar a linha debaixo
e ver os resultados na sequência.
