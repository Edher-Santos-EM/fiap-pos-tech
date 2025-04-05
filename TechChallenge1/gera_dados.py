import pandas as pd
import numpy as np

# Configurações iniciais
np.random.seed(42)  # Para resultados reproduzíveis
n = 150000  # Número de registros

# Gerar dados
dados = {
    'idade': np.random.randint(18, 80, n), #np.random.normal(loc=44, scale=10, size=n).clip(18, 80), 
    'gênero': np.random.choice(['masculino', 'feminino'], n, p=[0.5, 0.5]),
    'imc': np.round(np.random.normal(loc=28, scale=2, size=n).clip(18, 40), 2),  # IMC entre 18 e 40
    'filhos': np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.20, 0.25, 0.25, 0.15, 0.1, 0.05]),
    'fumante': np.random.choice(['sim', 'não'], n, p=[0.2, 0.8]),  # 20% fumantes
    'região': np.random.choice(['norte', 'nordeste', 'sudeste', 'sul', 'centro-oeste'], n, p=[0.05, 0.15, 0.50, 0.15, 0.15])
}

# Calcular encargos (base + ajustes por fatores)
df = pd.DataFrame(dados)

df['encargos'] = (
    15000  # Base
    + df['idade'] * 300  # Idade aumenta custo
    + df['imc'] * 400  # IMC aumenta custo
    + df['filhos'] * 500  # Filhos aumentam custo
    + np.where(df['fumante'] == 'sim', 8000, 0)  # Fumantes pagam mais
    + np.where(df['região'].isin(['sudeste', 'sul']), 2000, 0)  # Regiões mais caras
    + np.random.normal(0, 1000, n)  # Variação aleatória
).round(2)  # Arredondar para 2 casas decimais

# Como o algoritmo gera randomico, iremos tratar algumas informações para não aparecer "estranhices" como uma pessoa de 18 anos já ter 5 filhos
df = df[(df['idade'] - (df['filhos'] * 2) >= 16)] #Remover pessoas jovens com tantos filhos

# Salvar em CSV
df.to_csv('seguros.csv', index=False)