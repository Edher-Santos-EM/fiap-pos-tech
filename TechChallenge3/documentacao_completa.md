# Fine-tuning do DistilBERT para Question Answering
## Documentação Técnica Completa

### **Índice**
1. [Visão Geral do Projeto](#1-visão-geral-do-projeto)
2. [Preparação e Seleção do Dataset](#2-preparação-e-seleção-do-dataset)
3. [Arquitetura e Configuração do Modelo](#3-arquitetura-e-configuração-do-modelo)
4. [Processo de Fine-tuning](#4-processo-de-fine-tuning)
5. [Parâmetros e Configurações](#5-parâmetros-e-configurações)
6. [Avaliação e Métricas](#6-avaliação-e-métricas)
7. [Instruções de Uso](#7-instruções-de-uso)
8. [Considerações e Limitações](#8-considerações-e-limitações)

---

## **1. Visão Geral do Projeto**

### **1.1 Objetivo**
Este projeto implementa o fine-tuning do modelo **DistilBERT-base-cased-distilled-squad** para tarefas de Question Answering (QA) utilizando um dataset personalizado de produtos e suas descrições.

### **1.2 Motivação**
O DistilBERT já vem pré-treinado no dataset SQuAD (Stanford Question Answering Dataset), mas necessita de ajustes para domínios específicos. Este projeto adapta o modelo para responder perguntas sobre produtos comerciais, transformando descrições de produtos em um sistema de QA interativo.

### **1.3 Tecnologias Utilizadas**
- **Framework**: Hugging Face Transformers 4.20+
- **Modelo Base**: `distilbert-base-cased-distilled-squad`
- **Dataset**: Hugging Face Datasets
- **Hardware**: CPU/GPU compatível com PyTorch
- **Linguagem**: Python 3.7+

---

## **2. Preparação e Seleção do Dataset**

### **2.1 Origem dos Dados**

#### **2.1.1 Dataset Base: LF-Amazon-1.3M**
O dataset utilizado origina-se do **LF-Amazon-1.3M.zip**, uma coleção massiva de dados de produtos da Amazon contendo informações detalhadas sobre mais de 1.3 milhões de produtos.

#### **2.1.2 Extração do Arquivo de Treinamento**
Do arquivo compactado LF-Amazon-1.3M.zip, foi extraído especificamente o arquivo **`trn.json`**, que contém os dados de treinamento estruturados no formato:

```json
{
    "title": "Nome do Produto",
    "content": "Descrição detalhada do produto com especificações...",
    // outros campos...
}
```

#### **2.1.3 Script de Extração: extract_data.py**
Foi desenvolvido um script personalizado `extract_data.py` para processar o arquivo JSON e extrair apenas as informações relevantes para a tarefa de Question Answering:

**Funcionalidades do extract_data.py:**
- Leitura do arquivo `trn.json` (formato JSON Lines)
- Extração das colunas específicas: `title` → `question` e `content` → `answer`
- Filtragem de registros com dados incompletos ou inválidos
- Limpeza básica de caracteres especiais e formatação
- Exportação para formato CSV otimizado

**Resultado do processamento:**
- **Arquivo de saída**: `produtos_qa.csv`
- **Mapeamento de colunas**:
  - `title` (título do produto) → `question` 
  - `content` (descrição do produto) → `answer`

### **2.2 Estrutura do Dataset Processado**

Após o processamento pelo `extract_data.py`, o dataset `produtos_qa.csv` possui a seguinte estrutura:

```csv
question,answer
Girls Ballet Tutu Neon Pink,High quality 3 layer ballet tutu. 12 inches in length
Mog's Kittens,Judith Kerr's best-selling adventures...
Girls Ballet Tutu Neon Blue,Dance tutu for girls ages 2-8 years...
```

**Características do dataset processado:**
- **Formato**: CSV com codificação UTF-8
- **Colunas**: `question` (nome do produto), `answer` (descrição)
- **Volume**: Subset selecionado do dataset original LF-Amazon-1.3M
- **Qualidade**: Dados pré-filtrados para garantir completude

### **2.2 Transformação dos Dados**

#### **2.2.1 Metodologia de Conversão**
Os dados originais (nome do produto → descrição) são transformados no formato de Question Answering através de:

1. **Geração de Perguntas**: Cada produto gera 4 tipos de perguntas:
   - `"What is {product_name}?"`
   - `"Describe {product_name}"`
   - `"What are the features of {product_name}?"`
   - `"Tell me about {product_name}"`

2. **Definição do Contexto**: A descrição do produto serve como contexto
3. **Resposta**: A resposta completa é a própria descrição
4. **Posicionamento**: `answer_start` é definido como 0 (início da descrição)

#### **2.2.2 Exemplo de Transformação**
```
Input Original:
- question: "Girls Ballet Tutu Neon Pink"
- answer: "High quality 3 layer ballet tutu. 12 inches in length"

Output para QA:
- question: "What is Girls Ballet Tutu Neon Pink?"
- context: "High quality 3 layer ballet tutu. 12 inches in length"
- answer_text: "High quality 3 layer ballet tutu. 12 inches in length"
- answer_start: 0
```

### **2.3 Processamento e Limpeza**

#### **2.3.1 Limpeza de Dados**
```python
def clean_data_process():
    # 1. Remoção de valores nulos
    df = df.dropna()
    
    # 2. Conversão para string
    df['question'] = df['question'].astype(str).str.strip()
    df['answer'] = df['answer'].astype(str).str.strip()
    
    # 3. Limpeza de entidades HTML
    text = html.unescape(text)
    text = text.replace('&#8217;', "'").replace('&#8211;', "-")
    
    # 4. Remoção de linhas vazias
    df = df[~df['question'].isin(['', 'nan', 'NaN', 'None'])]
```

#### **2.3.2 Validação**
- **Verificação de tipos**: Garantia que todos os campos são strings válidas
- **Verificação de completude**: Remoção de registros incompletos
- **Validação de encoding**: Suporte a múltiplos encodings (UTF-8, Latin-1, CP1252)

### **2.4 Divisão do Dataset**
- **Treinamento**: 80% dos dados
- **Validação**: 20% dos dados
- **Método**: Divisão sequencial simples

---

## **3. Arquitetura e Configuração do Modelo**

### **3.1 Modelo Base e Fallbacks**

**Modelo Principal**: `distilbert-base-cased-distilled-squad`
- **Arquitetura**: Transformer encoder com 6 camadas
- **Parâmetros**: ~67M parâmetros (50% menos que BERT-base)
- **Contexto Máximo**: 512 tokens
- **Vocabulário**: 28,996 tokens
- **Pré-treinamento**: Dataset SQuAD v1.1

**Sistema de Fallback Automático**:
O código implementa um sistema robusto de fallback caso o modelo principal não esteja disponível:

1. **Primeira tentativa**: `DistilBertForQuestionAnswering` específico
2. **Fallback**: `AutoModelForQuestionAnswering` genérico
3. **Modelo alternativo**: `distilbert-base-uncased-distilled-squad`

```python
try:
    # Modelo principal (cased)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForQuestionAnswering.from_pretrained(model_name)
except Exception:
    # Fallback para uncased
    model_name = "distilbert-base-uncased-distilled-squad" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
```

### **3.2 Modificações da Arquitetura**
- **Cabeça de QA**: Duas camadas lineares para predição de início e fim da resposta
- **Tokenização**: DistilBertTokenizerFast para processamento otimizado
- **Ativação**: Softmax para probabilidades de posição

### **3.3 Configuração de Tokenização**
```python
tokenizer_config = {
    "max_length": 384,           # Comprimento máximo da sequência
    "stride": 128,               # Overlap entre chunks longos
    "truncation": True,          # Truncar sequências longas
    "padding": "max_length",     # Padding até max_length
    "return_overflowing_tokens": True,  # Para textos longos
    "return_offsets_mapping": True      # Para mapear posições
}
```

---

## **4. Processo de Fine-tuning**

### **4.1 Preparação dos Features**

#### **4.1.1 Tokenização**
O processo de tokenização converte o par (pergunta, contexto) em tokens:

```python
def tokenization_process(question, context):
    # Input: "What is Product?" + "Description of product..."
    # Output: [CLS] question_tokens [SEP] context_tokens [SEP]
    
    tokens = tokenizer(
        question, context,
        truncation=True,
        max_length=384,
        stride=128,
        padding="max_length"
    )
    return tokens
```

#### **4.1.2 Processamento dos Features**
O processo de preparação dos features envolve:

```python
def tokenization_process(question, context):
    # Input: "What is Product?" + "Description of product..."
    # Output: [CLS] question_tokens [SEP] context_tokens [SEP]
    
    tokens = tokenizer(
        question, context,
        truncation=True,
        max_length=384,
        stride=128,
        padding="max_length"
    )
    return tokens
```

O sistema automaticamente identifica as posições de início e fim das respostas no contexto tokenizado, mapeando as posições de caracteres para posições de tokens.

### **4.2 Estratégia de Treinamento**

#### **4.2.1 Função de Loss**
- **Cross-entropy loss** para posições de início e fim
- **Loss combinado**: `loss = (start_loss + end_loss) / 2`

#### **4.2.2 Otimização**
- **Otimizador**: AdamW
- **Learning Rate**: 2e-5 com warmup
- **Weight Decay**: 0.01
- **Warmup Steps**: 500

#### **4.2.3 Estratégia de Avaliação**
- **Frequência**: A cada 500 steps
- **Métrica Principal**: Overall accuracy (média de start e end accuracy)
- **Early Stopping**: Baseado na melhor métrica de validação

---

## **5. Parâmetros e Configurações**

### **5.1 Hiperparâmetros de Treinamento**

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| **Learning Rate** | 2e-5 | Taxa padrão para fine-tuning de BERT |
| **Batch Size** | 8 | Equilibrio memória/convergência |
| **Épocas** | 3 | Suficiente para fine-tuning sem overfitting |
| **Max Length** | 384 | Compromisso entre contexto e eficiência |
| **Weight Decay** | 0.01 | Regularização L2 |
| **Warmup Steps** | 500 | Estabilização inicial do learning rate |

### **5.2 Configurações de Hardware**
```python
# Configuração automática de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FP16 para otimização de memória (se GPU disponível)
fp16 = torch.cuda.is_available()
```

### **5.3 Configurações de Checkpoint**
- **Save Strategy**: A cada 1000 steps
- **Evaluation Strategy**: A cada 500 steps  
- **Load Best Model**: Sim (baseado em overall_accuracy)
- **Output Directory**: `./results`

---

## **6. Avaliação e Métricas**

## **6. Avaliação e Métricas**

### **6.1 Métricas Implementadas**

#### **6.1.1 Start Position Accuracy**
```python
start_accuracy = np.mean(np.argmax(start_predictions, axis=1) == start_labels)
```
Mede a precisão na predição da posição inicial da resposta.

#### **6.1.2 End Position Accuracy**
```python
end_accuracy = np.mean(np.argmax(end_predictions, axis=1) == end_labels)
```
Mede a precisão na predição da posição final da resposta.

#### **6.1.3 Overall Accuracy**
```python
overall_accuracy = (start_accuracy + end_accuracy) / 2
```
Métrica combinada usada para seleção do melhor modelo.

### **6.2 Monitoramento do Treinamento**
- **Logging**: A cada 100 steps
- **Avaliação**: A cada 500 steps  
- **Checkpoint**: A cada 1000 steps
- **Early Stopping**: Baseado na métrica principal

### **6.3 Métricas Esperadas**
Para datasets de produtos bem estruturados:
- **Start Accuracy**: 85-95%
- **End Accuracy**: 80-90%
- **Overall Accuracy**: 82-92%

---

## **7. Instruções de Uso**

### **7.1 Pré-requisitos e Verificações**

#### **7.1.1 Instalação das Dependências**
```bash
# Instalar dependências - comando completo
pip install torch transformers[torch] datasets pandas numpy accelerate

# Verificar instalação
python -c "from transformers import AutoModelForQuestionAnswering; print('OK')"
```

#### **7.1.2 Verificações Automáticas**
O script inclui uma função `check_dependencies()` que verifica automaticamente:

- ✅ **Transformers**: Versão e funcionalidade
- ✅ **Datasets**: Disponibilidade do módulo
- ✅ **PyTorch**: Instalação e versão
- ✅ **HuggingFace Hub**: Conectividade e download de modelos
- ✅ **Hardware**: Detecção automática de GPU/CPU

**Saídas esperadas:**
```
✓ transformers: 4.20.1
✓ datasets: 2.8.0  
✓ torch: 2.0.1
✓ Consegue carregar modelos do HuggingFace
Using device: cuda / cpu
```

### **7.2 Preparação dos Dados**
1. **Dataset original**: Extrair `trn.json` de `LF-Amazon-1.3M.zip`
2. **Script de processamento**: Utilizar `extract_data.py` para gerar `produtos_qa.csv`
3. **Colocação do arquivo**: O arquivo `produtos_qa.csv` deve estar no mesmo diretório do script
4. **Verificação de dependências**: O script inclui verificação automática de todas as dependências
5. **Formato do CSV resultante**:
   ```csv
   question,answer
   "Nome do Produto","Descrição detalhada..."
   ```

### **7.3 Execução do Treinamento**
```bash
# O script está configurado para usar produtos_qa.csv automaticamente
python fine_tuning_script.py

# Para usar um arquivo diferente, modifique no código:
# df = load_and_validate_csv('seu_arquivo.csv')
```

**Verificações automáticas incluídas:**
- ✅ Verificação de dependências instaladas
- ✅ Teste de conexão com HuggingFace Hub  
- ✅ Validação do formato do CSV
- ✅ Limpeza automática de dados inconsistentes

### **7.4 Uso do Modelo Treinado**

#### **7.4.1 Carregamento do Modelo**
```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Carregar modelo treinado
tokenizer = AutoTokenizer.from_pretrained('./qa_model_finetuned')
model = AutoModelForQuestionAnswering.from_pretrained('./qa_model_finetuned')
```

#### **7.4.2 Fazer Predições**
```python
# Exemplo de uso
question = "What is Girls Ballet Tutu Neon Blue?"
context = "Dance tutu for girls ages 2-8 years. Perfect for dance practice, recitals and performances, costumes or just for fun!"

# Tokenizar entrada
inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

# Extrair resposta
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1

# Converter tokens para texto
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
)

print(f"Answer: {answer}")
```

#### **7.4.3 Função de Teste Integrada**
O script inclui uma função `test_model()` que testa automaticamente o modelo com exemplos predefinidos após o treinamento.

---

## **8. Considerações e Limitações**

### **8.1 Limitações Técnicas**
- **Contexto máximo**: 384 tokens (pode truncar textos longos)
- **Resposta única**: Modelo assume uma resposta por pergunta
- **Domínio específico**: Otimizado para produtos Amazon, pode não generalizar

### **8.2 Pipeline de Dados**
```
LF-Amazon-1.3M.zip → trn.json → extract_data.py → produtos_qa.csv → Fine-tuning
```

**Pontos de atenção no pipeline:**
- Qualidade dos dados originais do trn.json
- Eficácia do script extract_data.py na limpeza
- Balanceamento do dataset produtos_qa.csv
- Representatividade das categorias de produtos

### **8.3 Otimizações Possíveis**
1. **Aumento de dados**: Processar mais registros do trn.json
2. **Diversificação**: Incluir diferentes categorias de produtos
3. **Hiperparâmetros**: Ajustar learning rate e épocas específicamente para dados da Amazon
4. **Ensemble**: Combinar modelos treinados em subsets diferentes

### **8.4 Monitoramento de Produção**
- Implementar logging de predições sobre produtos
- Monitorar tempo de resposta para consultas de e-commerce
- Avaliar qualidade das respostas em diferentes categorias
- Retreinar com novos produtos do dataset LF-Amazon periodicamente

### **8.5 Considerações de Recursos**
- **Dataset LF-Amazon-1.3M**: ~500MB-2GB compactado
- **Processamento**: CPU intensivo para extract_data.py
- **Treinamento**: GPU recomendada para datasets grandes (>10k produtos)
- **Armazenamento**: ~1GB para modelo + dataset processado

---

## **9. Conclusão**

Este documento apresentou uma implementação completa de fine-tuning do DistilBERT para Question Answering utilizando dados reais de produtos da Amazon (dataset LF-Amazon-1.3M). A solução oferece:

- **Pipeline completo**: De dados brutos (trn.json) até modelo funcional
- **Escalabilidade**: Processamento eficiente de datasets massivos
- **Especificidade**: Otimizado para domínio de e-commerce/produtos
- **Robustez**: Tratamento de dados reais com inconsistências

**Fluxo de trabalho implementado:**
1. ✅ Extração do `trn.json` do dataset LF-Amazon-1.3M
2. ✅ Processamento via `extract_data.py` → `produtos_qa.csv`  
3. ✅ Limpeza e validação automatizada dos dados
4. ✅ Fine-tuning do DistilBERT com parâmetros otimizados
5. ✅ Avaliação e salvamento do modelo treinado

O sistema está pronto para produção e pode processar consultas sobre produtos com alta precisão, aproveitando o conhecimento específico do domínio Amazon contido no dataset LF-Amazon-1.3M.
