import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando o arquivo CSV
def carregar_dados(caminho_arquivo):
    try:
        dados = pd.read_csv(caminho_arquivo)
        print(f"Dados carregados com sucesso. Formato: {dados.shape}")
        return dados
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return None

# Pré-processamento dos dados
def preprocessar_dados(dados):
    # Verificando valores ausentes
    print("Valores ausentes por coluna:")
    print(dados.isnull().sum())
    
    # Removendo valores ausentes (ou você pode optar por preencher)
    dados = dados.dropna()
    
    # Separando variáveis de entrada (X) e a variável alvo (y)
    # Considerando que todas as colunas exceto 'target' são características
    X = dados.drop('Target', axis=1)
    y = dados['Target']
    
    # Convertendo colunas categóricas para numéricas (excluindo a target)
    for coluna in X.select_dtypes(include=['object']).columns:
        X[coluna] = LabelEncoder().fit_transform(X[coluna])
    
    # Codificando a variável alvo
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Codificação da target: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Normalizando as características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, X.columns, le.classes_

# Divisão dos dados em treino e teste
def dividir_dados(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
    print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras")
    return X_train, X_test, y_train, y_test

# Treinamento do modelo de regressão logística
def treinar_modelo(X_train, y_train, max_iter=2000, random_state=42):
    modelo = LogisticRegression(max_iter=max_iter, random_state=random_state)
    modelo.fit(X_train, y_train)
    return modelo

# Avaliação do modelo
def avaliar_modelo(modelo, X_test, y_test, classes):
    # Fazendo previsões
    y_pred = modelo.predict(X_test)
    
    # Calculando acurácia
    acuracia = accuracy_score(y_test, y_pred)
    
    # Gerando relatório de classificação
    relatorio = classification_report(y_test, y_pred, target_names=classes)
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Acurácia do modelo: {acuracia:.4f}")
    print("\nRelatório de classificação:")
    print(relatorio)
    
    print("\nMatriz de confusão:")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão')
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.show()
    
    return acuracia, relatorio, cm

# Função para visualizar a importância das características
def visualizar_importancia_caracteristicas(modelo, caracteristicas):
    importancia = abs(modelo.coef_[0])
    indices = np.argsort(importancia)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Importância das Características')
    plt.bar(range(len(importancia)), importancia[indices], align='center')
    plt.xticks(range(len(importancia)), [caracteristicas[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Função para fazer previsões em novos dados
def prever_novos_dados(modelo, scaler, novos_dados, le):
    # Pré-processamento dos novos dados (mesmo processo usado no treinamento)
    for coluna in novos_dados.select_dtypes(include=['object']).columns:
        if coluna != 'target':  # Não codifique a coluna target se existir
            novos_dados[coluna] = LabelEncoder().fit_transform(novos_dados[coluna])
    
    # Aplicar a normalização
    novos_dados_scaled = scaler.transform(novos_dados)
    
    # Fazer previsões
    previsoes_num = modelo.predict(novos_dados_scaled)
    previsoes = le.inverse_transform(previsoes_num)
    
    return previsoes

# Salvar o modelo treinado
def salvar_modelo(modelo, caminho_arquivo='modelo_regressao_logistica.pkl'):
    import pickle
    with open(caminho_arquivo, 'wb') as arquivo:
        pickle.dump(modelo, arquivo)
    print(f"Modelo salvo em: {caminho_arquivo}")

# Carregar um modelo salvo
def carregar_modelo(caminho_arquivo='modelo_regressao_logistica.pkl'):
    import pickle
    with open(caminho_arquivo, 'rb') as arquivo:
        modelo = pickle.load(arquivo)
    print(f"Modelo carregado de: {caminho_arquivo}")
    return modelo

# Função principal que orquestra todo o processo
def main(caminho_arquivo, test_size=0.3, random_state=42, max_iter=2000):
    # Carregar dados
    dados = carregar_dados(caminho_arquivo)
    if dados is None:
        return
    
    # Explorar os dados
    print("\nPrimeiras 5 linhas dos dados:")
    print(dados.head())
    
    print("\nInformações dos dados:")
    print(dados.info())
    
    print("\nEstatísticas descritivas:")
    print(dados.describe())
    
    print("\nDistribuição da variável target:")
    print(dados['Target'].value_counts())
    
    # Pré-processar os dados
    X_scaled, y_encoded, caracteristicas, classes = preprocessar_dados(dados)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = dividir_dados(X_scaled, y_encoded, test_size, random_state)
    
    # Treinar o modelo
    modelo = treinar_modelo(X_train, y_train, max_iter, random_state)
    
    # Avaliar o modelo
    avaliar_modelo(modelo, X_test, y_test, classes)
    
    # Visualizar importância das características
    visualizar_importancia_caracteristicas(modelo, caracteristicas)
    
    # Salvar o modelo
    salvar_modelo(modelo)
    
    return modelo, caracteristicas, classes

# Executar o código principal
if __name__ == "__main__":
    # Substitua pelo caminho do seu arquivo CSV
    caminho_arquivo = "dropout-inaugural.csv"
    
    # Executar o processo completo
    modelo, caracteristicas, classes = main(caminho_arquivo)