"""
DistilBERT Question Answering Training Script

INSTALAÇÃO DAS DEPENDÊNCIAS:
Execute estes comandos antes de rodar o script:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers[torch]
pip install datasets
pip install pandas numpy
pip install accelerate

OU execute tudo de uma vez:
pip install torch transformers[torch] datasets pandas numpy accelerate

VERIFICAR INSTALAÇÃO:
python -c "from transformers import AutoModelForQuestionAnswering; print('✓ Transformers OK')"
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re

# Verificar e importar transformers
try:
    from transformers import (
        DistilBertTokenizerFast, 
        DistilBertForQuestionAnswering,
        TrainingArguments, 
        Trainer,
        default_data_collator
    )
    print("✓ DistilBertForQuestionAnswering importado com sucesso")
except ImportError:
    try:
        # Tentar importação alternativa
        from transformers import (
            DistilBertTokenizer as DistilBertTokenizerFast,
            AutoModelForQuestionAnswering,
            TrainingArguments, 
            Trainer,
            default_data_collator
        )
        # Usar AutoModel como fallback
        DistilBertForQuestionAnswering = AutoModelForQuestionAnswering
        print("✓ Usando AutoModelForQuestionAnswering como fallback")
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("Execute: pip install transformers[torch] datasets accelerate")
        raise

try:
    from datasets import Dataset as HFDataset
    print("✓ datasets importado com sucesso")
except ImportError:
    print("❌ Módulo 'datasets' não encontrado")
    print("Execute: pip install datasets")
    raise

# Configuração do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def check_dependencies():
    """
    Verifica se todas as dependências estão instaladas corretamente
    """
    print("Verificando dependências...")
    
    try:
        import transformers
        print(f"✓ transformers: {transformers.__version__}")
    except ImportError:
        print("❌ transformers não instalado")
        return False
    
    try:
        import datasets
        print(f"✓ datasets: {datasets.__version__}")
    except ImportError:
        print("❌ datasets não instalado")
        return False
    
    try:
        import torch
        print(f"✓ torch: {torch.__version__}")
    except ImportError:
        print("❌ torch não instalado")
        return False
    
    # Testar se consegue carregar um modelo
    try:
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
        print("✓ Consegue carregar modelos do HuggingFace")
        return True
    except Exception as e:
        print(f"❌ Erro ao carregar modelo de teste: {e}")
        print("Verifique sua conexão com a internet")
        return False

class QADataset:
    def __init__(self, csv_data):
        """
        Preprocessa os dados para o formato de Question Answering
        """
        self.data = []
        
        # Limpar dados primeiro
        csv_data = csv_data.dropna()  # Remove linhas com valores nulos
        csv_data = csv_data.reset_index(drop=True)
        
        for _, row in csv_data.iterrows():
            # Garantir que são strings e não estão vazias
            product_name = str(row['question']).strip()
            description = str(row['answer']).strip()
            
            # Pular se algum campo estiver vazio após limpeza
            if not product_name or not description or product_name == 'nan' or description == 'nan':
                continue
                
            # Limpar caracteres especiais HTML
            description = self._clean_html_entities(description)
            
            # Criar diferentes tipos de perguntas sobre o produto
            questions = [
                f"What is {product_name}?",
                f"Describe {product_name}",
                f"What are the features of {product_name}?",
                f"Tell me about {product_name}"
            ]
            
            # Para cada pergunta, usar a descrição como contexto e resposta
            for question in questions:
                # A resposta será a descrição completa
                answer_text = description
                
                # Encontrar onde a resposta começa no contexto (neste caso, no início)
                answer_start = 0
                
                # Garantir que todos os valores são strings
                qa_item = {
                    'question': str(question),
                    'context': str(description),
                    'answer_text': str(answer_text),
                    'answer_start': int(answer_start)
                }
                
                self.data.append(qa_item)
    
    def _clean_html_entities(self, text):
        """
        Limpa entidades HTML do texto
        """
        import html
        # Decodificar entidades HTML
        text = html.unescape(text)
        # Remover caracteres especiais problemáticos
        text = text.replace('&#8217;', "'").replace('&#8211;', "-").replace('&#8212;', "-")
        return text
    
    def get_data(self):
        return self.data

def prepare_train_features(examples, tokenizer, max_length=384):
    """
    Prepara os features para treinamento
    """
    # Tokenizar perguntas e contextos
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        max_length=max_length,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Mapear para os exemplos originais
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Preparar rótulos
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Pegar o exemplo original
        sample_index = sample_mapping[i]
        answer = examples["answer_text"][sample_index]
        start_char = examples["answer_start"][sample_index]
        end_char = start_char + len(answer)

        # Encontrar o token de início da sequência (após [CLS] e pergunta)
        sequence_ids = tokenized_examples.sequence_ids(i)
        
        # Encontrar onde o contexto começa
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Verificar se a resposta está completamente no contexto
        if not (offsets[token_start_index][0] <= start_char and 
                offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Encontrar tokens de início e fim da resposta
            while token_start_index < len(offsets) and offsets[token_start_index][0] < start_char:
                token_start_index += 1
            
            while offsets[token_end_index][1] > end_char:
                token_end_index -= 1

            tokenized_examples["start_positions"].append(token_start_index)
            tokenized_examples["end_positions"].append(token_end_index)

    return tokenized_examples

def compute_metrics(eval_predictions):
    """
    Métricas de avaliação simples
    """
    predictions, labels = eval_predictions
    start_predictions = predictions[0]
    end_predictions = predictions[1]
    
    start_labels = labels[0]
    end_labels = labels[1]
    
    # Accuracy simples para start e end positions
    start_accuracy = np.mean(np.argmax(start_predictions, axis=1) == start_labels)
    end_accuracy = np.mean(np.argmax(end_predictions, axis=1) == end_labels)
    
    return {
        "start_accuracy": start_accuracy,
        "end_accuracy": end_accuracy,
        "overall_accuracy": (start_accuracy + end_accuracy) / 2
    }

def load_and_validate_csv(file_path):
    """
    Carrega e valida dados de um arquivo CSV
    """
    try:
        # Tentar diferentes encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✓ CSV carregado com encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("Não foi possível carregar o CSV com nenhum encoding")
        
        # Verificar se tem as colunas necessárias
        required_columns = ['question', 'answer']
        if not all(col in df.columns for col in required_columns):
            available_cols = list(df.columns)
            print(f"❌ Colunas disponíveis: {available_cols}")
            print(f"❌ Colunas necessárias: {required_columns}")
            
            # Tentar mapear colunas automaticamente
            if len(available_cols) >= 2:
                print(f"Usando as duas primeiras colunas: {available_cols[0]} e {available_cols[1]}")
                df = df.rename(columns={
                    available_cols[0]: 'question',
                    available_cols[1]: 'answer'
                })
            else:
                raise ValueError("CSV deve ter pelo menos 2 colunas")
        
        # Remover linhas com valores nulos
        original_len = len(df)
        df = df.dropna(subset=['question', 'answer'])
        df = df.reset_index(drop=True)
        
        # Converter para string e limpar
        df['question'] = df['question'].astype(str).str.strip()
        df['answer'] = df['answer'].astype(str).str.strip()
        
        # Remover linhas vazias ou com 'nan'
        df = df[~df['question'].isin(['', 'nan', 'NaN', 'None'])]
        df = df[~df['answer'].isin(['', 'nan', 'NaN', 'None'])]
        df = df.reset_index(drop=True)
        
        print(f"✓ Dados limpos: {len(df)} linhas (era {original_len})")
        
        if len(df) == 0:
            raise ValueError("Nenhuma linha válida encontrada após limpeza")
        
        return df
        
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        raise

def validate_and_clean_data(qa_data):
    """
    Valida e limpa os dados antes de convertê-los para HuggingFace Dataset
    """
    cleaned_data = []
    
    for item in qa_data:
        # Verificar se todos os campos necessários existem
        if not all(key in item for key in ['question', 'context', 'answer_text', 'answer_start']):
            print(f"⚠️ Item incompleto ignorado: {item}")
            continue
        
        # Garantir tipos corretos
        try:
            cleaned_item = {
                'question': str(item['question']).strip(),
                'context': str(item['context']).strip(),
                'answer_text': str(item['answer_text']).strip(),
                'answer_start': int(item['answer_start'])
            }
            
            # Verificar se não há campos vazios
            if not cleaned_item['question'] or not cleaned_item['context'] or not cleaned_item['answer_text']:
                print(f"⚠️ Item com campos vazios ignorado")
                continue
                
            # Verificar se answer_start é válido
            if cleaned_item['answer_start'] < 0:
                cleaned_item['answer_start'] = 0
            
            cleaned_data.append(cleaned_item)
            
        except Exception as e:
            print(f"⚠️ Erro ao processar item: {e}")
            continue
    
    print(f"✓ {len(cleaned_data)} itens válidos de {len(qa_data)} originais")
    return cleaned_data
    """
    Valida e limpa os dados antes de convertê-los para HuggingFace Dataset
    """
    cleaned_data = []
    
    for item in qa_data:
        # Verificar se todos os campos necessários existem
        if not all(key in item for key in ['question', 'context', 'answer_text', 'answer_start']):
            print(f"⚠️ Item incompleto ignorado: {item}")
            continue
        
        # Garantir tipos corretos
        try:
            cleaned_item = {
                'question': str(item['question']).strip(),
                'context': str(item['context']).strip(),
                'answer_text': str(item['answer_text']).strip(),
                'answer_start': int(item['answer_start'])
            }
            
            # Verificar se não há campos vazios
            if not cleaned_item['question'] or not cleaned_item['context'] or not cleaned_item['answer_text']:
                print(f"⚠️ Item com campos vazios ignorado")
                continue
                
            # Verificar se answer_start é válido
            if cleaned_item['answer_start'] < 0:
                cleaned_item['answer_start'] = 0
            
            cleaned_data.append(cleaned_item)
            
        except Exception as e:
            print(f"⚠️ Erro ao processar item: {e}")
            continue
    
    print(f"✓ {len(cleaned_data)} itens válidos de {len(qa_data)} originais")
    return cleaned_data

def train_qa_model():
    # Carregar dados
    print("Carregando dados...")
    
    # Para usar seu CSV, descomente e modifique esta linha:
    df = load_and_validate_csv('produtos_qa.csv')
    
    # Exemplo de dados para teste (remova isso quando usar seu CSV)
    # data = {
    #     'question': [
    #         'Girls Ballet Tutu Neon Pink',
    #         'Mog\'s Kittens',
    #         'Girls Ballet Tutu Neon Blue'
    #     ],
    #     'answer': [
    #         'High quality 3 layer ballet tutu. 12 inches in length',
    #         'Judith Kerr best-selling adventures of that endearing (and exasperating) cat Mog have entertained children for more than 30 years. Now, even infants and toddlers can enjoy meeting this loveable feline. These sturdy little board books with their bright, simple pictures, easy text, and hand-friendly formats are just the thing to delight the very young. Ages 6 months-2 years.',
    #         'Dance tutu for girls ages 2-8 years. Perfect for dance practice, recitals and performances, costumes or just for fun!'
    #     ]
    # }
    
    # df = pd.DataFrame(data)
    
    # Verificar dados carregados
    print(f"Dados originais carregados: {len(df)} linhas")
    print("Primeiras linhas:")
    print(df.head())
    
    # Verificar se há valores nulos
    print("\nVerificando valores nulos:")
    print(df.isnull().sum())
    
    # Preparar dataset
    qa_dataset = QADataset(df)
    qa_data = qa_dataset.get_data()
    
    # Validar e limpar dados
    qa_data = validate_and_clean_data(qa_data)
    
    if len(qa_data) == 0:
        raise ValueError("❌ Nenhum dado válido encontrado após limpeza!")
    
    print(f"Total de exemplos criados: {len(qa_data)}")
    
    # Mostrar exemplo dos dados processados
    print("\nExemplo de dados processados:")
    print(f"Question: {qa_data[0]['question']}")
    print(f"Context: {qa_data[0]['context'][:100]}...")
    print(f"Answer: {qa_data[0]['answer_text'][:50]}...")
    
    # Dividir dados em treino e validação
    split_idx = max(1, int(0.8 * len(qa_data)))  # Garantir pelo menos 1 item para validação
    train_data = qa_data[:split_idx]
    val_data = qa_data[split_idx:]
    
    print(f"Dados de treino: {len(train_data)}")
    print(f"Dados de validação: {len(val_data)}")
    
    # Converter para formato de datasets do Hugging Face com tratamento de erro
    try:
        print("Convertendo dados para HuggingFace Dataset...")
        train_dataset = HFDataset.from_list(train_data)
        val_dataset = HFDataset.from_list(val_data)
        print("✓ Datasets criados com sucesso")
    except Exception as e:
        print(f"❌ Erro ao criar HuggingFace Dataset: {e}")
        print("Verificando tipos de dados...")
        for i, item in enumerate(train_data[:3]):
            print(f"Item {i}: {[(k, type(v)) for k, v in item.items()]}")
        raise
    
    # Carregar tokenizer e modelo
    print("Carregando modelo e tokenizer...")
    model_name = "distilbert-base-cased-distilled-squad"
    
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        model = DistilBertForQuestionAnswering.from_pretrained(model_name)
        print(f"✓ Modelo {model_name} carregado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao carregar {model_name}: {e}")
        print("Tentando usar modelo alternativo...")
        try:
            # Fallback para modelo genérico
            from transformers import AutoTokenizer, AutoModelForQuestionAnswering
            model_name = "distilbert-base-uncased-distilled-squad"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            print(f"✓ Modelo alternativo {model_name} carregado com sucesso")
        except Exception as e2:
            print(f"❌ Erro ao carregar modelo alternativo: {e2}")
            raise
    
    # Preprocessar dados
    print("Preprocessando dados...")
    train_dataset = train_dataset.map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    val_dataset = val_dataset.map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    
    # Configurar argumentos de treinamento
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",  # Mudança aqui
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
        fp16=torch.cuda.is_available(),  # Use FP16 se GPU disponível
    )
    
    # Criar trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Treinar modelo
    print("Iniciando treinamento...")
    trainer.train()
    
    # Salvar modelo
    print("Salvando modelo...")
    trainer.save_model("./qa_model_finetuned")
    tokenizer.save_pretrained("./qa_model_finetuned")
    
    return model, tokenizer

def test_model(model, tokenizer):
    """
    Testar o modelo treinado
    """
    print("\nTestando modelo...")
    
    # Exemplo de teste
    question = "What is Girls Ballet Tutu Neon Pink?"
    context = "High quality 3 layer ballet tutu. 12 inches in length"
    
    # Tokenizar
    inputs = tokenizer(question, context, return_tensors="pt")
    
    # Fazer predição
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obter tokens de resposta
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    
    # Encontrar posições de início e fim da resposta
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    
    # Converter para texto
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    # Verificar dependências primeiro
    if not check_dependencies():
        print("\n❌ Algumas dependências não estão instaladas corretamente.")
        print("Execute os comandos de instalação no topo do arquivo.")
        exit(1)
    
    # Treinar modelo
    model, tokenizer = train_qa_model()
    
    # Testar modelo
    test_model(model, tokenizer)
    
    print("\nTreinamento concluído! Modelo salvo em './qa_model_finetuned'")
    
    # Exemplo de como carregar e usar o modelo salvo
    print("\nPara usar o modelo salvo:")
    print("""
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    import torch
    
    # Carregar modelo
    tokenizer = AutoTokenizer.from_pretrained('./qa_model_finetuned')
    model = AutoModelForQuestionAnswering.from_pretrained('./qa_model_finetuned')
    
    # Fazer predição
    question = "What is Girls Ballet Tutu Neon Blue?"
    context = "Dance tutu for girls ages 2-8 years. Perfect for dance practice, recitals and performances, costumes or just for fun!"
    
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    print(f"Answer: {answer}")
    """)
    
    print("\n" + "="*50)
    print("COMO USAR SEU PRÓPRIO CSV:")
    print("="*50)
    print("""
    1. Coloque seu arquivo CSV no mesmo diretório do script
    2. Modifique a linha no código:
       # df = load_and_validate_csv('seu_arquivo.csv')
       df = load_and_validate_csv('meu_dataset.csv')  # ← descomente e use seu arquivo
    
    3. Comente ou remova a seção de dados de exemplo:
       # data = {...}  # ← comente isso
       # df = pd.DataFrame(data)  # ← e isso também
    
    4. Seu CSV deve ter duas colunas com nomes 'question' e 'answer'
       Ou simplesmente duas colunas (serão renomeadas automaticamente)
    
    Exemplo de CSV válido:
    question,answer
    "Product Name 1","Description of product 1"
    "Product Name 2","Description of product 2"
    """)