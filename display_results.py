from IPython.display import Markdown

def resultados_caracteres(title: str, correct: dict, wrong: dict, elements: dict):
    print("RESULTADOS "+title)
    header = "|Caractere| Acerto        | Erro          |\n"
    line = "|:--------|---------------:|:---------------|\n"
    lines = []
    for letter in ['6', '7', 'a', 'b', 'c', 'd', 'h', 'm', 'n', 'x']:
        lines.append("|"+letter+" |"+str(correct[letter])+"/"+str(elements[letter])+" ("+"{0:.2f}%".format((correct[letter]/elements[letter])*100)+")|"+str(wrong[letter])+"/"+str(elements[letter])+" ("+"{0:.2f}%".format((wrong[letter]/elements[letter])*100) +")|"   )
    lines = "\n".join(lines)
    text = header+line+lines
    return (Markdown(text))


def resultados_acuracia(title: str, info: list):
    print("RESULTADOS "+title)
    header = "| Métrica                    | Taxa  |\n"
    line = "|:-:|:-:|\n"
    lines = []
    for l in info:
        lines.append("|"+l[0]+"      | "+l[1]+"%|")
    lines = "\n".join(lines)
    text = header+line+lines
    return (Markdown(text))
