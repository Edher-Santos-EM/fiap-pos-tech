# Tech Challenge 4 - Análise de Vídeo com IA

Sistema de análise de vídeo usando Deep Learning para:
- 🎬 Detecção de cenas
- 😊 Análise de emoções (DeepFace)
- 🏃 Detecção de atividades (YOLO + VideoMAE)

---

## 🚀 Instalação Rápida

```batch
setup_environment.bat
```

📖 **[Guia Completo de Instalação](INSTALACAO.md)**

---

## ▶️ Uso

1. Coloque seu vídeo como `video.mp4` na pasta raiz
2. Execute:
```bash
python run_pipeline.py
```

---

## 📋 Requisitos

- Python 3.11 (obrigatório)
- GPU NVIDIA (opcional, mas recomendado)

---

## 📁 Resultados

Os resultados são salvos em `output/`:
- `output/cenas/` - Cenas detectadas
- `output/sentimentos/` - Análise de emoções
- `output/atividades/` - Atividades detectadas

---

## 🆘 Ajuda

Veja o **[Guia de Instalação](INSTALACAO.md)** para:
- Instalar Python 3.11
- Configurar GPU
- Resolver problemas comuns

---

**Tecnologias**: TensorFlow, PyTorch, DeepFace, Ultralytics YOLO, OpenCV
