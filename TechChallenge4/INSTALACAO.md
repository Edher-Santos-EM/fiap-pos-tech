# Guia de Instalação

## 🚀 Instalação Rápida

### Windows

```batch
setup_environment.bat
```

### Linux / macOS

```bash
python3 setup_environment.py
```

---

## ⚙️ O que o script faz?

1. ✅ Detecta Python 3.11 no sistema
2. ✅ Detecta GPU NVIDIA (se disponível)
3. ✅ Cria ambiente virtual
4. ✅ Instala todas as dependências
5. ✅ Configura GPU automaticamente
6. ✅ Testa a instalação

---

## ❓ Python 3.11 não encontrado?

O script mostrará um menu com opções:

### **[M] Caminho Manual**
Se você já tem Python 3.11 instalado:
```
Digite o caminho completo:
C:\Python311\python.exe
```

### **[L] Localizar**
Não sabe onde está instalado? Execute:
```batch
find_python.bat
```
Isso mostrará todos os Pythons no seu PC.

### **[D] Download Direto** (Recomendado)
Baixa automaticamente Python 3.11.9:
- ✅ [Download Windows 64-bit](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
- ⚠️ **IMPORTANTE**: Marque "Add Python 3.11 to PATH" ao instalar

### **[P] Página de Downloads**
Abre a página oficial para escolher manualmente.

---

## 🐧 Instalar Python 3.11 no Linux

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### Fedora/RHEL
```bash
sudo dnf install python3.11
```

### Arch Linux
```bash
sudo pacman -S python311
```

---

## 🍎 Instalar Python 3.11 no macOS

### Usando Homebrew
```bash
brew install python@3.11
```

---

## ✅ Verificar Instalação

Após executar o script, verifique:

```bash
python test_gpu.py
```

**Saída esperada com GPU:**
```
[1] TensorFlow:
   OK Versao: 2.17.1
   Built with CUDA: True
   GPUs disponiveis: 1

[2] Pillow/PIL: OK
[3] DeepFace: OK
[4] NVIDIA GPU: NVIDIA RTX A4500
```

**Saída sem GPU (CPU):**
```
[1] TensorFlow:
   OK Versao: 2.17.1
   Built with CUDA: False
   GPUs disponiveis: 0
```

---

## 🎬 Usar o Pipeline

### 1. Ativar ambiente virtual

**Windows:**
```batch
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 2. Colocar vídeo

Coloque seu vídeo como `video.mp4` na pasta raiz do projeto.

### 3. Executar pipeline

```bash
python run_pipeline.py
```

---

## 🐛 Problemas Comuns

### "Python 3.11 não encontrado"

**Solução 1**: Localizar no sistema
```batch
find_python.bat
```

**Solução 2**: Baixar e instalar
```
setup_environment.bat → [D] Download
```

### "GPU não detectada" (mas tenho GPU NVIDIA)

**Causa**: Python 3.12+ não tem suporte completo ao TensorFlow GPU

**Solução**: Use Python 3.11 especificamente

### "Comando python não reconhecido"

**Windows**: Python não está no PATH
- Reinstale marcando "Add to PATH"
- Ou use opção [M] com caminho completo

**Linux/Mac**: Use `python3.11` em vez de `python`

---

## 📊 Performance

| Hardware | Análise de Emoções | Detecção de Atividades | Status |
|----------|-------------------|------------------------|--------|
| GPU NVIDIA | ~5-10x mais rápido | ~3-5x mais rápido | Requer TensorFlow GPU |
| CPU | Funcional | Pode demorar mais | ✅ Funciona sempre |

**Nota**: O sistema funciona perfeitamente com CPU. GPU é opcional e melhora a velocidade.

---

## 📁 Estrutura de Arquivos

```
TechChallenge4/
├── setup_environment.bat    # Script de instalação (Windows)
├── setup_environment.py     # Script de instalação (Linux/Mac)
├── find_python.bat          # Localiza Python no sistema
├── test_gpu.py              # Testa configuração
├── run_pipeline.py          # Executa análise
├── video.mp4                # SEU VÍDEO AQUI
├── venv/                    # Ambiente virtual (criado pelo script)
└── output/                  # Resultados (criado automaticamente)
    ├── cenas/
    ├── sentimentos/
    └── atividades/
```

---

## ℹ️ Requisitos

### Obrigatório
- **Python 3.11** (não use 3.12 ou 3.13)

### Opcional (mas recomendado)
- GPU NVIDIA com drivers atualizados
- CUDA 12.x (instalado automaticamente pelo script)

---

## 🆘 Precisa de Ajuda?

1. Execute `python test_gpu.py` e compartilhe a saída
2. Verifique se está usando Python 3.11: `python --version`
3. Reinstale o ambiente: execute `setup_environment.bat` novamente

---

**Por que Python 3.11?**

TensorFlow com suporte a GPU requer Python 3.11. Versões 3.12+ ainda não têm suporte completo.

---

**Última atualização**: Novembro 2025
