import json
import pandas as pd
import random
from typing import List, Dict, Any

def carregar_json_lines(arquivo_path: str) -> List[Dict]:
    """Carrega arquivo JSONL (uma linha = um objeto JSON)"""
    produtos = []
    
    try:
        with open(arquivo_path, 'r', encoding='utf-8') as file:
            for linha_num, linha in enumerate(file, 1):
                linha = linha.strip()
                if linha:  # Ignora linhas vazias
                    try:
                        produto = json.loads(linha)
                        produtos.append(produto)
                    except json.JSONDecodeError as e:
                        print(f"Erro na linha {linha_num}: {e}")
                        continue
        
        print(f"Carregados {len(produtos)} produtos do arquivo {arquivo_path}")
        return produtos
        
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {arquivo_path}")
        return []
    except Exception as e:
        print(f"Erro ao carregar arquivo: {e}")
        return []
    
produtos = carregar_json_lines('trn.json')

produtos_qa = []
for produto in produtos:
    if len(produto['content']) > 0 and len(produto['title']) > 0:
        qa_format = {
            'question': produto['title'],
            'answer': produto['content']
        }        
        produtos_qa.append(qa_format)

df = pd.DataFrame(produtos_qa)
df.to_csv('produtos_qa.csv', index=False, encoding='utf-8')