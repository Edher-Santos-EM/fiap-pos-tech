# Algoritmos Genéticos para Distribuição Equilibrada de Alunos em Turmas

## Definição do Problema
O problema de distribuição equilibrada de alunos em turmas é uma variante complexa dos problemas de otimização educacional que visa distribuir estudantes em grupos de forma a maximizar o equilíbrio em múltiplos critérios simultaneamente. Este problema pode ser classificado como um problema de otimização multi-objetivo com restrições, pertencente à classe NP-difícil.

## Características do Problema:

* Variáveis de decisão: Atribuição de cada aluno a uma turma específica
* Objetivos múltiplos: Equilibrar distribuição considerando quatro critérios principais:
    - Sexo/Gênero: Distribuição equilibrada entre masculino e feminino
    - Idade: Balanceamento etário para diversidade geracional nas turmas
    - Primeira letra do nome: Distribuição alfabética para evitar concentrações
    - Escola de origem: Integração de alunos de diferentes instituições
* Restrições rígidas: Capacidade máxima das turmas
* Restrições flexiveis: Proporção ideal de alunos nas turmas

##  Por que Algoritmos Genéticos?
* Capacidade de lidar com múltiplos objetivos conflitantes
* Flexibilidade para incorporar diferentes tipos de restrições sejam elas rígidas como capacidade ou flexíveis como proporções ideais
* Robustez em espaços de busca complexos onde não existe solução "perfeita". Ex: 43 Meninas e 76 Meninos não permite distribuição 50/50
* Habilidade de encontrar soluções "suficientemente boas" que evitam extremos indesejáveis