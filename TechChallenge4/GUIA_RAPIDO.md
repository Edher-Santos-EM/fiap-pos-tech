# 🚀 Guia Rápido - Execução de Etapas

## 📋 Resumo dos Ambientes

| Etapa | Ambiente | Frameworks | Tempo |
|-------|----------|-----------|-------|
| **1. Cenas** | `venv_scenes` | OpenCV, SceneDetect | Rápido |
| **2. Emoções** | `venv_emotions` | TensorFlow, DeepFace, MediaPipe | Médio |
| **3. Atividades** | `venv_activities` | PyTorch, YOLO, VideoMAE | Médio-Lento |

---

## ⚡ Execução Rápida

### Pipeline Completo (Automático)

```bash
# Coloque o vídeo em: videos/video.mp4
python run_pipeline.py
```

O script alterna automaticamente entre os 3 ambientes! ✨

---

## 🎯 Execução Individual das Etapas

### Etapa 1: Detectar Cenas 🎬

**Ambiente:** `venv_scenes`

```bash
# Windows
venv_scenes\Scripts\python.exe src/cli/etapa1_separar_cenas.py --input videos/video.mp4

# Linux/Mac
venv_scenes/bin/python src/cli/etapa1_separar_cenas.py --input videos/video.mp4
```

**Resultado:**
- ✅ `output/cenas/cena_001.mp4`, `cena_002.mp4`, ...
- ✅ `output/cenas/thumbnails/` - Miniaturas
- ✅ `output/cenas/relatorio_cenas.md` - Relatório

---

### Etapa 2: Analisar Emoções 😊

**Ambiente:** `venv_emotions`

```bash
# Windows
venv_emotions\Scripts\python.exe src/cli/etapa2_analisar_sentimentos.py

# Linux/Mac
venv_emotions/bin/python src/cli/etapa2_analisar_sentimentos.py
```

**Resultado:**
- ✅ `output/sentimentos/cena_001_sentimentos.mp4`, ...
- ✅ `output/sentimentos/relatorio_sentimentos.md`
- ✅ Vídeos com emoções anotadas (Feliz, Triste, Raiva, etc.)

---

### Etapa 3: Detectar Atividades 🏃

**Ambiente:** `venv_activities`

#### 3a. Híbrido (Recomendado) 🔀

Combina VideoMAE (dinâmicas) + YOLO Pose (estáticas)

```bash
# Windows
venv_activities\Scripts\python.exe src/cli/etapa3_hibrido.py

# Linux/Mac
venv_activities/bin/python src/cli/etapa3_hibrido.py
```

**Detecta:**
- 🤖 VideoMAE: Dançando, Acenando, Caretas, Gargalhadas
- 🎯 Pose: Trabalhando (laptop), Lendo (livro), Telefone

---

#### 3b. VideoMAE Puro 🤖

```bash
# Windows
venv_activities\Scripts\python.exe src/cli/etapa3_videomae.py

# Linux/Mac
venv_activities/bin/python src/cli/etapa3_videomae.py
```

**Detecta:** Apenas atividades dinâmicas

---

#### 3c. Análise de Pose 🎯

```bash
# Windows
venv_activities\Scripts\python.exe src/cli/etapa3_interpretar_atividades.py

# Linux/Mac
venv_activities/bin/python src/cli/etapa3_interpretar_atividades.py
```

**Detecta:** Apenas atividades com objetos (laptop, livro, telefone)

---

## 🔧 Opções Úteis

### Usar CPU ao invés de GPU

```bash
# Etapa 2
venv_emotions\Scripts\python.exe src/cli/etapa2_analisar_sentimentos.py --device cpu

# Etapa 3
venv_activities\Scripts\python.exe src/cli/etapa3_hibrido.py --device cpu
```

### Processar Cenas Específicas

```bash
# Criar pasta com cenas selecionadas
mkdir output\cenas_selecionadas
copy output\cenas\cena_001.mp4 output\cenas_selecionadas\
copy output\cenas\cena_005.mp4 output\cenas_selecionadas\

# Processar apenas essas
venv_emotions\Scripts\python.exe src/cli/etapa2_analisar_sentimentos.py --input-dir output/cenas_selecionadas
```

### Ajustar Sensibilidade de Detecção de Cenas

```bash
# Mais sensível (mais cenas)
venv_scenes\Scripts\python.exe src/cli/etapa1_separar_cenas.py --threshold 20.0

# Menos sensível (menos cenas)
venv_scenes\Scripts\python.exe src/cli/etapa1_separar_cenas.py --threshold 35.0
```

### Ajustar Confiança das Detecções

```bash
# Método híbrido com confiança customizada
venv_activities\Scripts\python.exe src/cli/etapa3_hibrido.py --videomae-conf 0.4 --pose-conf 0.6
```

---

## 📊 Exemplo Completo Passo a Passo

```bash
# 1. Colocar vídeo
copy "C:\Videos\meu_filme.mp4" videos\video.mp4

# 2. Detectar cenas (venv_scenes)
venv_scenes\Scripts\python.exe src/cli/etapa1_separar_cenas.py

# 3. Analisar emoções (venv_emotions)
venv_emotions\Scripts\python.exe src/cli/etapa2_analisar_sentimentos.py

# 4. Detectar atividades - método híbrido (venv_activities)
venv_activities\Scripts\python.exe src/cli/etapa3_hibrido.py

# 5. Ver resultados
explorer output\hibrido
```

---

## 🆘 Solução de Problemas

### "ModuleNotFoundError"

**Problema:** Ambiente errado ativado

**Solução:** Cada etapa precisa do seu ambiente:
- Etapa 1 → `venv_scenes`
- Etapa 2 → `venv_emotions`
- Etapa 3 → `venv_activities`

### "CUDA out of memory"

```bash
# Use CPU ao invés de GPU
--device cpu

# Ou processe vídeos menores
ffmpeg -i video.mp4 -vf scale=640:360 video_small.mp4
```

### Ambientes não existem

```bash
# Criar todos os ambientes
python setup_dual_environments.py

# Ou apenas o ambiente de cenas (rápido)
python setup_scenes_only.py
```

---

## 📁 Estrutura de Saída

```
output/
├── cenas/                      # Etapa 1
│   ├── cena_001.mp4
│   ├── cena_002.mp4
│   ├── thumbnails/
│   └── relatorio_cenas.md
│
├── sentimentos/                # Etapa 2
│   ├── cena_001_sentimentos.mp4
│   ├── cena_002_sentimentos.mp4
│   └── relatorio_sentimentos.md
│
└── hibrido/                    # Etapa 3 (híbrido)
    ├── cena_001_hibrido.mp4
    ├── cena_002_hibrido.mp4
    └── relatorio_hibrido.md
```

---

## 💡 Dicas de Performance

| Situação | Recomendação |
|----------|--------------|
| GPU NVIDIA disponível | Use `--device auto` (padrão) |
| Sem GPU | Use `--device cpu` |
| Vídeo muito grande | Reduza resolução com `ffmpeg` |
| Processamento lento | Reduza `--fps` na Etapa 2 |
| Muitas cenas detectadas | Aumente `--threshold` na Etapa 1 |
| Poucas cenas detectadas | Diminua `--threshold` na Etapa 1 |

---

## 🔗 Links Úteis

- **README completo:** [README.md](README.md)
- **Documentação de ambientes:** Seção "Ambientes Virtuais" no README
- **Troubleshooting:** Seção "Troubleshooting" no README
