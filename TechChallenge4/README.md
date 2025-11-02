# 🎬 Sistema de Análise de Vídeo - Tech Challenge 4

Sistema completo para análise automatizada de vídeos com detecção de cenas, emoções e atividades humanas usando Deep Learning.

---

## ⚡ Início Rápido (3 Passos)

```bash
# 1. Criar ambientes virtuais (15-25 min)
python setup_dual_environments.py

# 2. Colocar seu vídeo
# Copie para: videos/video.mp4

# 3. Executar pipeline completo
python run_pipeline.py
```

**Requisitos:** Python 3.11 | GPU NVIDIA recomendada (opcional)

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Arquitetura](#-arquitetura)
- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Uso Rápido](#-uso-rápido)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Pipeline Completo](#-pipeline-completo)
- [Ambientes Virtuais](#-ambientes-virtuais)
- [Modelos de IA](#-modelos-de-ia)
- [Configuração Avançada](#-configuração-avançada)
- [Troubleshooting](#-troubleshooting)
- [Desenvolvimento](#-desenvolvimento)

---

## 🎯 Visão Geral

Este projeto analisa vídeos em três etapas:

### Etapa 1: Detecção de Cenas
- Segmenta o vídeo automaticamente em cenas distintas
- Detecta mudanças de conteúdo e transições
- Exporta cada cena como arquivo separado

### Etapa 2: Análise de Sentimentos
- Detecta faces usando MediaPipe e DeepFace
- Classifica emoções: Feliz, Triste, Raiva, Surpreso, Neutro, Medo, Nojo
- Gera vídeos anotados com emoções e relatórios detalhados

### Etapa 3: Interpretação de Atividades
Três métodos disponíveis:

**🔀 Híbrido (Recomendado)**
- VideoMAE para atividades dinâmicas: Dançando, Acenando, Fazendo Caretas
- YOLO Pose para atividades estáticas: Trabalhando, Lendo, Telefone
- Combina o melhor de cada método

**🤖 VideoMAE**
- Modelo de transformer para reconhecimento de ações
- Ótimo para movimentos complexos

**🎯 Análise de Pose (YOLO)**
- Detecção de pose + objetos
- Ótimo para atividades com objetos específicos

---

## 🏗️ Arquitetura

O projeto utiliza **3 ambientes virtuais isolados** para evitar conflitos de dependências:

```
┌─────────────────────────────────────────────────────┐
│                  Vídeo de Entrada                    │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │   ETAPA 1: Cenas      │
         │   venv_scenes         │
         │   • OpenCV            │
         │   • SceneDetect       │
         │   • NumPy, SciPy      │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ ETAPA 2: Emoções      │
         │ venv_emotions         │
         │ • TensorFlow 2.20+    │
         │ • DeepFace            │
         │ • MediaPipe           │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ ETAPA 3: Atividades   │
         │ venv_activities       │
         │ • PyTorch             │
         │ • VideoMAE            │
         │ • YOLO11              │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  Vídeos + Relatórios  │
         └───────────────────────┘
```

---

## 💻 Requisitos

### Sistema Operacional
- ✅ Windows 10/11
- ✅ Linux (Ubuntu 20.04+)
- ✅ macOS (Intel/Apple Silicon)

### Software Obrigatório
- **Python 3.11** (obrigatório - não use 3.12 ou 3.13)
- **Git** (para clonar o repositório)

### Hardware Recomendado
- **GPU NVIDIA** com suporte CUDA (altamente recomendado)
  - CUDA 11.x ou 12.x
  - 6+ GB VRAM
  - Drivers NVIDIA atualizados
- **CPU**: 8+ cores (alternativa sem GPU, muito mais lento)
- **RAM**: 16+ GB
- **Disco**: 15+ GB livres (modelos + ambientes)

### Hardware Mínimo
- CPU: 4 cores
- RAM: 8 GB
- Disco: 10 GB

> ⚠️ **Nota**: Sem GPU, o processamento será 10-30x mais lento.

---

## 🚀 Instalação

> ⚠️ **IMPORTANTE**: Siga os passos nesta ordem exata!

### 1. Clonar o Repositório

```bash
git clone <url-do-repositorio>
cd TechChallenge4
```

### 2. Verificar/Instalar Python 3.11

**Verificar se já tem:**
```bash
python --version
# Deve mostrar: Python 3.11.x
```

**Se não tiver Python 3.11:**

#### Windows
```bash
# Download do instalador
https://www.python.org/downloads/release/python-3119/

# Durante instalação:
# ✅ Marcar "Add Python 3.11 to PATH"
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

#### macOS
```bash
brew install python@3.11
```

### 3. Configurar Ambientes Virtuais (OBRIGATÓRIO)

O projeto usa **três ambientes virtuais separados** para evitar conflitos de dependências.

#### Instalação Automática (Recomendado)

```bash
python setup_dual_environments.py
```

**O que o script faz:**
1. ✅ Detecta Python 3.11 automaticamente
2. ✅ Identifica sua GPU NVIDIA (modelo, driver, versão CUDA)
3. ✅ Cria `venv_scenes` com OpenCV + SceneDetect (leve, 2-3 min)
4. ✅ Cria `venv_emotions` com TensorFlow + DeepFace + MediaPipe
5. ✅ Cria `venv_activities` com PyTorch + VideoMAE + YOLO
6. ✅ Instala pacotes CUDA corretos (11.x ou 12.x) baseado na sua GPU
7. ✅ Testa os três ambientes e confirma funcionamento da GPU

**Durante a instalação:**
- Pressione `Enter` ou `s` quando solicitado
- Se ambientes já existirem, escolha se quer recriar
- Downloads: ~4-5 GB de pacotes
- Tempo total: 15-25 minutos (depende da conexão)

**Saída esperada:**
```
[1/8] Verificando Python 3.11...
    OK: py -3.11
    Versão: Python 3.11.9

[2/8] Detectando GPU NVIDIA e CUDA...
    OK: GPU NVIDIA detectada
    Modelo: NVIDIA GeForce RTX 3080
    Driver: 537.13
    CUDA suportado: 12.2
    → Instalando pacotes para CUDA 12.x

[3-6/8] Criando ambientes e instalando pacotes...
    ✅ venv_emotions criado
    ✅ venv_activities criado

[7/8] Testando ambientes...
    ✅ TensorFlow: 2.20.0 | GPUs: 1
    ✅ PyTorch: 2.5.1+cu121 | CUDA: True

[8/8] INSTALAÇÃO CONCLUÍDA!
```

#### Instalação Manual

Se preferir controle total:

```bash
# Ambiente 1: Emoções
python3.11 -m venv venv_emotions

# Windows
venv_emotions\Scripts\activate
pip install -r requirements_emotions.txt

# Linux/Mac
source venv_emotions/bin/activate
pip install -r requirements_emotions.txt

# Ambiente 2: Atividades
python3.11 -m venv venv_activities

# Windows
venv_activities\Scripts\activate
pip install -r requirements_activities.txt

# Linux/Mac
source venv_activities/bin/activate
pip install -r requirements_activities.txt
```

### 4. Verificar Instalação (Opcional)

**Windows - Menu Interativo:**
```bash
manage_envs.bat
# Opção 4: Testar ambos os ambientes
```

**Manual - Testar TensorFlow:**
```bash
venv_emotions\Scripts\python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPU: {tf.config.list_physical_devices(\"GPU\")}')"
```

**Manual - Testar PyTorch:**
```bash
venv_activities\Scripts\python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**✅ Tudo OK se mostrar:**
- Versões instaladas (TensorFlow 2.20+, PyTorch 2.0+)
- GPU detectada (se você tiver NVIDIA)

---

## ⚡ Uso Rápido

> 💡 **Dica**: Após instalar os ambientes, tudo é automático!

### Pipeline Completo (Recomendado)

1. **Coloque seu vídeo** em `videos/video.mp4`
   ```bash
   # Copie seu vídeo para a pasta videos
   # Exemplo: videos/video.mp4
   ```

2. **Execute o pipeline**:
   ```bash
   python run_pipeline.py
   ```

   O script irá:
   - ✅ Verificar se ambientes existem
   - ✅ Alternar automaticamente entre ambientes conforme necessário
   - ✅ Executar as 3 etapas sequencialmente

3. **Escolha o método** para Etapa 3:
   - `1` - Híbrido (recomendado)
   - `2` - VideoMAE
   - `3` - Análise de Pose

4. **Resultados** estarão em:
   - `output/cenas/` - Cenas detectadas
   - `output/sentimentos/` - Análise de emoções
   - `output/hibrido/` ou `output/videomae/` - Atividades

### Executar Etapas Individualmente

> 💡 **Importante**: Cada etapa usa seu próprio ambiente virtual!

#### Etapa 1: Detecção de Cenas

Usa: `venv_scenes` (leve, sem deep learning)

```bash
# Windows
venv_scenes\Scripts\python.exe src/cli/etapa1_separar_cenas.py --input videos/video.mp4

# Linux/Mac
venv_scenes/bin/python src/cli/etapa1_separar_cenas.py --input videos/video.mp4
```

**Opções disponíveis:**
```bash
--input         # Vídeo de entrada (padrão: videos/video.mp4)
--output        # Pasta de saída (padrão: output/cenas)
--threshold     # Sensibilidade de detecção (padrão: 27.0)
--min-duration  # Duração mínima da cena em segundos (padrão: 1.0)
```

**Saída:**
- `output/cenas/cena_001.mp4`, `cena_002.mp4`, etc.
- `output/cenas/thumbnails/` - Miniaturas das cenas
- `output/cenas/cenas_metadata.json` - Metadados
- `output/cenas/relatorio_cenas.md` - Relatório

---

#### Etapa 2: Análise de Sentimentos

Usa: `venv_emotions` (TensorFlow + DeepFace + MediaPipe)

```bash
# Windows
venv_emotions\Scripts\python.exe src/cli/etapa2_analisar_sentimentos.py

# Linux/Mac
venv_emotions/bin/python src/cli/etapa2_analisar_sentimentos.py
```

**Opções disponíveis:**
```bash
--input-dir     # Pasta com cenas (padrão: output/cenas)
--output-dir    # Pasta de saída (padrão: output/sentimentos)
--device        # Dispositivo: cuda, cpu, auto (padrão: auto)
--fps           # Frames por segundo a analisar (padrão: 6)
```

**Saída:**
- `output/sentimentos/cena_001_sentimentos.mp4`, etc.
- `output/sentimentos/relatorio_sentimentos.md` - Relatório detalhado
- Vídeos anotados com emoções detectadas

---

#### Etapa 3: Análise de Atividades

Usa: `venv_activities` (PyTorch + YOLO + VideoMAE)

**3a. Método Híbrido (Recomendado)**

Combina VideoMAE para atividades dinâmicas e YOLO Pose para estáticas:

```bash
# Windows
venv_activities\Scripts\python.exe src/cli/etapa3_hibrido.py

# Linux/Mac
venv_activities/bin/python src/cli/etapa3_hibrido.py
```

**Opções disponíveis:**
```bash
--input-dir          # Pasta com cenas (padrão: output/cenas)
--output-dir         # Pasta de saída (padrão: output/hibrido)
--device             # Dispositivo: cuda, cpu, auto (padrão: auto)
--videomae-conf      # Confiança VideoMAE (padrão: 0.3)
--pose-conf          # Confiança Pose (padrão: 0.5)
--pose-model         # Modelo YOLO Pose (padrão: models/yolo11x-pose.pt)
--object-model       # Modelo YOLO Objetos (padrão: models/yolo11x.pt)
```

**Detecta:**
- 🤖 VideoMAE: Dançando, Acenando, Caretas, Gargalhadas
- 🎯 Pose+Objetos: Trabalhando (laptop), Lendo (livro), Telefone

**Saída:**
- `output/hibrido/cena_001_hibrido.mp4`, etc.
- `output/hibrido/relatorio_hibrido.md` - Relatório

---

**3b. VideoMAE Puro**

Apenas modelo de IA para atividades dinâmicas:

```bash
# Windows
venv_activities\Scripts\python.exe src/cli/etapa3_videomae.py

# Linux/Mac
venv_activities/bin/python src/cli/etapa3_videomae.py
```

**Detecta:** Dançando, Acenando, Caretas, Gargalhadas, etc.

---

**3c. Análise de Pose Pura**

Apenas YOLO Pose + detecção de objetos:

```bash
# Windows
venv_activities\Scripts\python.exe src/cli/etapa3_interpretar_atividades.py

# Linux/Mac
venv_activities/bin/python src/cli/etapa3_interpretar_atividades.py
```

**Detecta:** Trabalhando, Lendo, Telefone

---

### Exemplos de Uso Completo

**Processar vídeo específico do início ao fim:**
```bash
# 1. Detectar cenas
venv_scenes\Scripts\python.exe src/cli/etapa1_separar_cenas.py --input videos/meu_video.mp4

# 2. Analisar emoções
venv_emotions\Scripts\python.exe src/cli/etapa2_analisar_sentimentos.py

# 3. Detectar atividades (híbrido)
venv_activities\Scripts\python.exe src/cli/etapa3_hibrido.py
```

**Processar apenas algumas cenas específicas:**
```bash
# Copie manualmente as cenas desejadas para uma pasta
mkdir output\cenas_selecionadas
copy output\cenas\cena_001.mp4 output\cenas_selecionadas\
copy output\cenas\cena_005.mp4 output\cenas_selecionadas\

# Processe apenas essas cenas
venv_emotions\Scripts\python.exe src/cli/etapa2_analisar_sentimentos.py --input-dir output/cenas_selecionadas
```

**Usar CPU ao invés de GPU:**
```bash
venv_emotions\Scripts\python.exe src/cli/etapa2_analisar_sentimentos.py --device cpu
venv_activities\Scripts\python.exe src/cli/etapa3_hibrido.py --device cpu
```

---

## 📁 Estrutura do Projeto

```
TechChallenge4/
│
├── 📄 README.md                          # Este arquivo
├── 📄 setup_dual_environments.py         # Setup automático
├── 📄 run_pipeline.py                    # Pipeline completo
├── 📄 manage_envs.bat                    # Gerenciador (Windows)
│
├── 📋 requirements_emotions.txt          # Deps para emoções
├── 📋 requirements_activities.txt        # Deps para atividades
│
├── 🗂️ venv_emotions/                     # Ambiente virtual 1
│   └── TensorFlow + DeepFace + MediaPipe
│
├── 🗂️ venv_activities/                   # Ambiente virtual 2
│   └── PyTorch + Transformers + YOLO
│
├── 🗂️ models/                            # Modelos YOLO
│   ├── yolo11x.pt                       # Detecção de objetos
│   └── yolo11x-pose.pt                  # Detecção de pose
│
├── 🗂️ src/                               # Código-fonte
│   ├── analyzers/                       # Analisadores
│   │   ├── emotion_analyzer.py          # DeepFace + MediaPipe
│   │   ├── activity_analyzer.py         # YOLO Pose
│   │   ├── videomae_analyzer.py         # VideoMAE
│   │   └── hybrid_analyzer.py           # Combinado
│   │
│   ├── activities/                      # Detectores de atividades
│   │   ├── reading_activity.py
│   │   ├── phone_activity.py
│   │   ├── working_activity.py
│   │   ├── dancing_activity.py
│   │   └── ...
│   │
│   ├── cli/                             # Scripts de linha de comando
│   │   ├── etapa1_separar_cenas.py
│   │   ├── etapa2_analisar_sentimentos.py
│   │   ├── etapa3_interpretar_atividades.py
│   │   ├── etapa3_videomae.py
│   │   └── etapa3_hibrido.py
│   │
│   ├── core/                            # Núcleo do sistema
│   │   ├── video_processor.py
│   │   └── report_generator.py
│   │
│   └── utils/                           # Utilitários
│       ├── config.py
│       ├── logger.py
│       └── progress_bar.py
│
└── 🗂️ output/                            # Resultados
    ├── cenas/                           # Etapa 1
    ├── sentimentos/                     # Etapa 2
    ├── atividades/                      # Etapa 3 (Pose)
    ├── videomae/                        # Etapa 3 (VideoMAE)
    └── hibrido/                         # Etapa 3 (Híbrido)
```

---

## 🔄 Pipeline Completo

### Como Funciona

O `run_pipeline.py` executa todas as etapas sequencialmente, **alternando automaticamente entre os ambientes virtuais**:

```python
# Pseudo-código do pipeline
video.mp4
    ↓
[venv_emotions] Etapa 1: Separar Cenas
    ↓ output/cenas/cena_*.mp4
[venv_emotions] Etapa 2: Analisar Sentimentos
    ↓ output/sentimentos/
[venv_activities] Etapa 3: Analisar Atividades
    ↓ output/hibrido/ (ou videomae/ ou atividades/)
```

### Opções de Linha de Comando

#### Etapa 1
```bash
python src/cli/etapa1_separar_cenas.py --help

Opções:
  --input, -i          Vídeo de entrada (padrão: video.mp4)
  --output-dir, -o     Diretório de saída (padrão: output/cenas)
  --threshold, -t      Threshold de detecção (padrão: 25.0)
  --min-duration, -m   Duração mínima da cena em segundos (padrão: 1.0)
```

#### Etapa 2
```bash
python src/cli/etapa2_analisar_sentimentos.py --help

Opções:
  --input-dir, -i      Diretório com cenas (padrão: output/cenas)
  --output-dir, -o     Diretório de saída (padrão: output/sentimentos)
  --confidence, -c     Confiança mínima (padrão: 0.5)
  --device, -d         Dispositivo: auto/cuda/cpu (padrão: auto)
  --no-scores          Não mostrar scores de confiança
```

#### Etapa 3 (Híbrido)
```bash
python src/cli/etapa3_hibrido.py --help

Opções:
  --input-dir, -i      Diretório com cenas (padrão: output/cenas)
  --output-dir, -o     Diretório de saída (padrão: output/hibrido)
  --device, -d         Dispositivo: auto/cuda/cpu (padrão: auto)
  --clip-duration      Duração dos clips VideoMAE (padrão: 2.0s)
  --overlap            Overlap entre clips (padrão: 1.0s)
```

---

## 🔀 Ambientes Virtuais

### Por que Três Ambientes?

O projeto utiliza três frameworks com dependências conflitantes:
- **Etapa 1**: OpenCV + SceneDetect (leve, sem deep learning)
- **Etapa 2**: TensorFlow 2.20+ (para análise de emoções)
- **Etapa 3**: PyTorch (para análise de atividades)

**Conflitos principais:**
- TensorFlow e PyTorch têm versões incompatíveis de `protobuf`
- Conflitos em bibliotecas CUDA entre os frameworks
- Competição por recursos da GPU
- Etapa 1 não precisa do overhead de frameworks de deep learning

**Solução**: Isolar cada etapa em seu próprio ambiente virtual.

### Estrutura dos Ambientes

#### 1️⃣ venv_scenes (Detecção de Cenas)
```
Propósito: Etapa 1 - Separação de cenas

Dependências principais:
├── opencv-python >= 4.8.0
├── scenedetect[opencv] >= 0.6.0
├── numpy >= 1.24.0
├── scipy >= 1.10.0
└── pillow >= 10.0.0

Usado por:
└── src/cli/etapa1_separar_cenas.py

Tempo de instalação: ~2-3 minutos
```

#### 2️⃣ venv_emotions (Análise de Sentimentos)
```
Propósito: Etapa 2 - Detecção de emoções

Dependências principais:
├── tensorflow >= 2.20.0
├── deepface >= 0.0.79
├── mediapipe >= 0.10.0
├── tf-keras >= 2.20.0
└── opencv-python, numpy, pillow...

Usado por:
└── src/cli/etapa2_analisar_sentimentos.py

Tempo de instalação: ~5-8 minutos
```

#### 3️⃣ venv_activities (Análise de Atividades)
```
Propósito: Etapa 3 - Detecção de atividades

Dependências principais:
├── torch >= 2.0.0
├── transformers >= 4.57.0
├── ultralytics >= 8.0.0 (YOLO11)
├── protobuf >= 6.33.0
└── opencv-python, numpy, pillow...

Usado por:
├── src/cli/etapa3_hibrido.py
├── src/cli/etapa3_videomae.py
└── src/cli/etapa3_interpretar_atividades.py

Tempo de instalação: ~8-12 minutos
```

### Gerenciamento (Windows)

Use o `manage_envs.bat` para facilitar:

```bash
manage_envs.bat
```

Menu interativo:
```
[1] Ativar venv_emotions (para Etapa 2)
[2] Ativar venv_activities (para Etapa 3)
[3] Executar pipeline completo
[4] Testar ambos os ambientes
[5] Verificar versões instaladas
[6] Reinstalar ambientes
[S] Sair
```

### Comparação

| Aspecto         | Ambiente Único | Ambientes Duais |
|-----------------|----------------|-----------------|
| Conflitos       | ❌ Frequentes  | ✅ Nenhum       |
| Instalação      | ❌ Complexa    | ✅ Simples      |
| Manutenção      | ❌ Difícil     | ✅ Fácil        |
| Uso de disco    | ✅ ~5 GB       | ⚠️ ~8 GB        |
| Confiabilidade  | ⚠️ Instável    | ✅ Estável      |

---

## 🤖 Modelos de IA

### Modelos Utilizados

#### DeepFace (Emoções)
```
Localização: Baixado automaticamente por DeepFace
Diretório: ~/.deepface/weights/

Modelos:
├── retinaface.h5           # Detecção de faces
├── facial_expression_model_weights.h5  # Emoções
└── Outros modelos de classificação

Tamanho total: ~200 MB
```

#### MediaPipe (Detecção de Faces)
```
Localização: Instalado via pip
Modelos embutidos no pacote mediapipe

Funcionalidade:
└── Detecção rápida de faces e landmarks
```

#### YOLO11 (Pose e Objetos)
```
Localização: models/

Modelos:
├── yolo11x.pt              # Detecção de objetos (91 classes COCO)
├── yolo11x-pose.pt         # Detecção de pose (17 keypoints)

Tamanho: ~200 MB cada
Download: Automático na primeira execução
```

**Classes YOLO detectadas**:
- Pessoas
- Objetos: laptop, livro, celular, mouse, teclado
- 91 classes do dataset COCO

#### VideoMAE (Reconhecimento de Ações)
```
Localização: Baixado automaticamente por Transformers
Modelo: MCG-NJU/videomae-base-finetuned-kinetics

Tamanho: ~350 MB
Download: Automático na primeira execução

Atividades detectadas:
- 400 classes de ações (Kinetics-400)
- Selecionadas: Dançando, Acenando, Fazendo Caretas, etc.
```

### Configuração de Modelos

Edite `src/utils/config.py`:

```python
DEFAULT_CONFIG = {
    'activity_analysis': {
        'pose_model': 'models/yolo11x-pose.pt',  # Modelo de pose
        'object_model': 'models/yolo11x.pt',     # Modelo de objetos
        'confidence_threshold': 0.6,
    }
}
```

Ou via linha de comando:

```bash
python src/cli/etapa3_interpretar_atividades.py \
    --pose-model models/yolo11x-pose.pt \
    --object-model models/yolo11x.pt \
    --confidence 0.7
```

### Download Manual de Modelos

Se precisar baixar modelos manualmente:

```bash
# Ativar ambiente
venv_activities\Scripts\activate

# Baixar YOLO11
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt'); YOLO('yolo11x-pose.pt')"

# Mover para pasta models (se necessário)
move yolo11x*.pt models/  # Windows
mv yolo11x*.pt models/    # Linux/Mac
```

---

## ⚙️ Configuração Avançada

### Configuração de GPU/CUDA

O `setup_dual_environments.py` detecta automaticamente:

```
GPU Detectada: NVIDIA GeForce RTX 3080
Driver: 537.13
CUDA: 12.2

→ Instalando:
  • TensorFlow 2.20+ com nvidia-cudnn-cu12
  • PyTorch com cu121 (CUDA 12.1)
```

**CUDA 11.x**:
```bash
# TensorFlow
pip install tensorflow>=2.20.0
pip install nvidia-cudnn-cu11 nvidia-cublas-cu11

# PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.x**:
```bash
# TensorFlow
pip install tensorflow>=2.20.0
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12

# PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Sem GPU (CPU)**:
```bash
# TensorFlow
pip install tensorflow>=2.20.0

# PyTorch
pip install torch torchvision
```

### Ajustar Performance

#### Para GPUs com Pouca VRAM (<6 GB)

Edite os scripts CLI e reduza batch sizes:

```python
# Em etapa2_analisar_sentimentos.py
result = analyzer.process_scene(
    str(scene_path),
    str(output_path),
    batch_size=8  # Reduzir de 16 para 8
)

# Em etapa3_videomae.py
result = analyzer.process_scene(
    str(scene_path),
    str(output_path),
    batch_size=4  # Reduzir de 8 para 4
)
```

#### Para CPUs

```bash
# Usar menos workers
export NUM_WORKERS=2  # Linux/Mac
set NUM_WORKERS=2     # Windows

# Desabilitar half-precision
# (Adicione ao código se necessário)
```

---

## 🐛 Troubleshooting

### Problemas Comuns

#### 1. "Ambientes virtuais não encontrados"

**Erro:**
```
⚠️ AMBIENTES VIRTUAIS NÃO ENCONTRADOS
Execute: python setup_dual_environments.py
```

**Solução:**
```bash
# Você precisa criar os ambientes primeiro
python setup_dual_environments.py
```

**Causa:** Os ambientes `venv_scenes`, `venv_emotions` e `venv_activities` não foram criados ainda.

#### 2. "CUDA out of memory"

**Causa**: GPU sem memória suficiente.

**Soluções**:
```bash
# Opção 1: Processar vídeos menores
ffmpeg -i video.mp4 -vf scale=640:360 video_small.mp4

# Opção 2: Usar CPU
python src/cli/etapa2_analisar_sentimentos.py --device cpu

# Opção 3: Reduzir batch size (editar código)
```

#### 3. "TensorFlow não detecta GPU"

**Verificar**:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**Soluções**:
```bash
# Verificar driver NVIDIA
nvidia-smi

# Reinstalar bibliotecas CUDA
pip uninstall nvidia-cudnn-cu12 nvidia-cublas-cu12
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12

# Verificar versão TensorFlow
pip install tensorflow>=2.20.0
```

#### 4. "PyTorch não detecta GPU"

**Verificar**:
```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

**Soluções**:
```bash
# Reinstalar PyTorch com CUDA correto
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 5. "ModuleNotFoundError"

**Causa**: Ambiente errado ativado.

**Solução**:
```bash
# Para emoções
venv_emotions\Scripts\activate  # Windows
source venv_emotions/bin/activate  # Linux/Mac

# Para atividades
venv_activities\Scripts\activate  # Windows
source venv_activities/bin/activate  # Linux/Mac
```

#### 6. "Modelos YOLO não encontrados"

**Solução**:
```bash
# Verificar pasta models
dir models\*.pt  # Windows
ls models/*.pt   # Linux/Mac

# Se vazia, YOLO baixará automaticamente na primeira execução
# Os modelos serão salvos em models/
```

---

## 👨‍💻 Desenvolvimento

### Adicionar Nova Atividade

1. Crie arquivo `src/activities/minha_atividade.py`:
```python
from .base_activity import BaseActivity

class MinhaAtividade(BaseActivity):
    def detect(self, pose_data, object_data):
        # Sua lógica de detecção
        if self._check_conditions(pose_data, object_data):
            return True, 0.8  # confidence
        return False, 0.0

    def _check_conditions(self, pose_data, object_data):
        # Implementar verificações
        pass
```

2. Registre em `src/activities/__init__.py`:
```python
from .minha_atividade import MinhaAtividade
```

3. Adicione ao analisador em `src/analyzers/activity_analyzer.py`:
```python
self.activity_detectors = [
    MinhaAtividade(confidence_threshold),
    # ... outros detectores
]
```

---

## 📊 Benchmarks

### Performance (GPU NVIDIA RTX 3080)

| Etapa | Tempo (1 min vídeo) | VRAM Usada |
|-------|---------------------|------------|
| Etapa 1: Cenas | ~5s | N/A |
| Etapa 2: Emoções | ~30s | 2-3 GB |
| Etapa 3: Híbrido | ~60s | 4-5 GB |
| **Total** | **~95s** | **5 GB** |

### Performance (CPU Intel i7-10700K)

| Etapa | Tempo (1 min vídeo) |
|-------|---------------------|
| Etapa 1: Cenas | ~10s |
| Etapa 2: Emoções | ~8 min |
| Etapa 3: Híbrido | ~15 min |
| **Total** | **~23 min** |

> 📝 **Nota**: Tempos variam conforme complexidade do vídeo (número de pessoas, movimentos, etc.)

---

## 📚 Referências

### Frameworks e Bibliotecas

- **TensorFlow**: https://www.tensorflow.org/
- **PyTorch**: https://pytorch.org/
- **DeepFace**: https://github.com/serengil/deepface
- **MediaPipe**: https://google.github.io/mediapipe/
- **Ultralytics YOLO**: https://docs.ultralytics.com/
- **Transformers (Hugging Face)**: https://huggingface.co/docs/transformers/
- **VideoMAE**: https://github.com/MCG-NJU/VideoMAE

---

## 🔄 Atualizações Recentes

### v2.0.0 (2025-01-02)
- ✨ Implementado sistema de ambientes virtuais duais
- ✨ Detecção automática de GPU e versão CUDA
- ✨ Suporte a YOLO11 (melhor precisão)
- ✨ Método híbrido VideoMAE + Pose
- ✨ Modelos YOLO organizados em pasta `models/`
- 🐛 Corrigidos conflitos de dependências
- 📚 Documentação consolidada em README único

---

**Tech Challenge 4 - FIAP Pós-Tech**

Desenvolvido com ❤️ para análise inteligente de vídeos
