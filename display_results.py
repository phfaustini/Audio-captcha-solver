from IPython.display import Markdown


def resultados_caracteres(title: str, correct: dict, wrong: dict, elements: dict):
    header = "### "+title + "\n|Caractere| Acerto        | Erro          |\n"
    line = "|:--------|---------------:|:---------------|\n"
    lines = []
    for letter in ['6', '7', 'a', 'b', 'c', 'd', 'h', 'm', 'n', 'x']:
        tot = elements[letter]
        if tot > 0:
            cr = 100 * correct[letter]/tot
            wr = 100 * wrong[letter]/tot
        else:
            cr = wr = 0
        lines.append("| %s | %d/%d (%.2f) | %d/%d (%.2f)" % (letter, correct[letter], tot, cr, wrong[letter], tot, wr ))
    lines = "\n".join(lines)
    text = header+line+lines
    return (Markdown(text))


def resultados_acuracia(title: str, info: list):
    print()
    header = "#### Acurácia "+title + "\n| Métrica | Taxa  |\n"
    line = "|:-:|:-:|\n"
    lines = []
    for arr in info:
        for l in arr:
            lines.append("|"+l[0]+"| "+l[1]+"%|")
    lines = "\n".join(lines)
    text = header+line+lines
    return (Markdown(text))
