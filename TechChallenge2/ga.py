from collections import Counter


def calcula_penalidade(turma, geral):
    penalidade = 0
    for k, v in turma.items():
        piso = geral[k][0]
        teto = geral[k][1]        
        if v < piso:
            penalidade += (piso - v)
        elif v > teto:
            penalidade += (v - teto)
    return penalidade

def eh_aluno(lista_completa_alunos, index):
    return not lista_completa_alunos[index] is None;

def obtenha_alunos(lista_completa_alunos, turma):
    return [lista_completa_alunos[i] for i in turma if not lista_completa_alunos[i] is None]


