# Documentação do Processo de Fine-Tuning para Question Answering

## 1. Seleção e Preparação do Dataset

### 1.1. Origem dos Dados
O dataset utilizado foi extraído do arquivo `trn.json` presente no conjunto de dados LF-Amazon-1.3M.zip. Este arquivo contém informações sobre produtos da Amazon, incluindo título e conteúdo (descrição).

### 1.2. Extração e Transformação
Foi desenvolvido um programa em Python chamado `extract_data.py` para processar o arquivo `trn.json` e extrair as colunas `title` e `content`, transformando-as em um novo dataset no formato CSV com as colunas `question` e `answer`. O novo dataset foi salvo como `produtos_qa.csv`.

**Trecho do código de extração (exemplo):**
```python
import json
import pandas as pd

# Carrega o arquivo trn.json
with open('trn.json', 'r') as f:
    data = [json.loads(line) for line in f]

# Extrai as colunas 'title' e 'content'
df = pd.DataFrame(data)
df = df[['title', 'content']]

# Renomeia as colunas para 'question' e 'answer'
df.columns = ['question', 'answer']

# Salva como CSV
df.to_csv('produtos_qa.csv', index=False)
```

### 1.3. Carregamento e Validação
O código principal inclui uma função `load_and_validate_csv` que carrega o arquivo CSV e valida sua estrutura. A função:
- Tenta diferentes encodings para ler o arquivo (utf-8, latin-1, cp1252, iso-8859-1)
- Verifica se as colunas `question` e `answer` estão presentes
- Remove linhas com valores nulos
- Converte as colunas para string e aplica limpeza básica

**Trecho do código de validação:**
```python
def load_and_validate_csv(file_path):
    # Tentar diferentes encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    # Verificar colunas necessárias
    if not all(col in df.columns for col in ['question', 'answer']):
        # Mapear automaticamente as duas primeiras colunas
        df = df.rename(columns={df.columns[0]: 'question', df.columns[1]: 'answer'})
```

### 1.4. Pré-processamento para QA
A classe `QADataset` transforma os dados do CSV em exemplos de question answering:
- Para cada produto, gera múltiplas perguntas variadas
- Usa a descrição do produto como contexto e resposta
- Limpa entidades HTML e caracteres especiais

**Exemplo de transformação:**
```python
# Para o produto "Girls Ballet Tutu Neon Pink"
questions = [
    "What is Girls Ballet Tutu Neon Pink?",
    "Describe Girls Ballet Tutu Neon Pink",
    "What are the features of Girls Ballet Tutu Neon Pink?",
    "Tell me about Girls Ballet Tutu Neon Pink"
]
```

### 1.5. Validação e Limpeza
A função `validate_and_clean_data` garante a qualidade dos dados:
- Verifica a presença de todos os campos obrigatórios
- Converte tipos de dados para os formatos corretos
- Remove itens com campos vazios ou inválidos

## 2. Fine-Tuning do Modelo

### 2.1. Modelo Base
O modelo utilizado para fine-tuning é o `distilbert-base-cased-distilled-squad`, uma versão do DistilBert fine-tuned no dataset SQuAD.

### 2.2. Tokenização e Preparação dos Dados
A função `prepare_train_features` prepara os dados para o modelo:
- Tokeniza perguntas e contextos com truncamento (max_length=384)
- Utiliza stride=128 para lidar com contextos longos
- Mapeia as posições de início e fim das respostas

**Parâmetros de tokenização:**
```python
tokenizer(
    examples["question"],
    examples["context"],
    truncation=True,
    max_length=384,
    stride=128,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)
```

### 2.3. Divisão dos Dados
Os dados são divididos em:
- Treino: 80% dos exemplos
- Validação: 20% dos exemplos

### 2.4. Parâmetros de Treinamento
Os hiperparâmetros configurados para o fine-tuning:

**TrainingArguments:**
```python
TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="overall_accuracy",
    greater_is_better=True,
    warmup_steps=500,
    fp16=torch.cuda.is_available(),
)
```

### 2.5. Métricas de Avaliação
A função `compute_metrics` calcula:
- Acurácia das posições de início da resposta
- Acurácia das posições de fim da resposta
- Acurácia geral (média das duas)

### 2.6. Processo de Treinamento
O treinamento é realizado usando a classe `Trainer` do Hugging Face, que gerencia:
- Loop de treinamento com otimização
- Avaliação periódica no conjunto de validação
- Salvamento do melhor modelo
- Logging do progresso

### 2.7. Salvamento do Modelo
Após o treinamento, o modelo e tokenizer são salvos no diretório `./qa_model_finetuned` para uso futuro.

## 3. Resultados e Uso

### 3.1. Teste do Modelo
O modelo treinado pode ser testado com exemplos específicos:

**Exemplo de uso:**
```python
question = "What is Girls Ballet Tutu Neon Pink?"
context = "High quality 3 layer ballet tutu. 12 inches in length"

# Tokenizar entrada
inputs = tokenizer(question, context, return_tensors="pt")

# Fazer predição
outputs = model(**inputs)

# Extrair resposta
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
)
```

### 3.2. Considerações Finais
O processo de fine-tuning adaptou o modelo DistilBert para o domínio específico de produtos da Amazon, permitindo que ele responda perguntas sobre características e descrições de produtos com base em seu contexto.

O modelo resultante pode ser integrado em sistemas de atendimento ao cliente, catálogos de produtos ou qualquer aplicação que necessite de respostas precisas sobre informações de produtos.