"""Gerenciamento de configurações."""

import yaml
import torch
from pathlib import Path
from typing import Any, Dict


class Config:
    """Gerencia configurações do sistema."""

    DEFAULT_CONFIG = {
        'device': 'auto',
        'verbose': True,
        'scene_detection': {
            'threshold': 25.0,
            'min_duration': 1.0,
            'codec': 'libx264',
            'quality': 23
        },
        'emotion_analysis': {
            'confidence_threshold': 0.5,
            'show_scores': True,
            # DeepFace usa RetinaFace para detecção (baixado automaticamente)
            # e modelos próprios para classificação de emoções
        },
        'activity_analysis': {
            'confidence_threshold': 0.6,
            'show_skeleton': True,
            'pose_model': 'models/yolov8n-pose.pt',
            'object_model': 'models/yolov8n.pt'
        },
        'performance': {
            'batch_size': 16,
            'use_half_precision': True,
            'num_workers': 4
        }
    }

    def __init__(self, config_path: str = None):
        """
        Inicializa configuração.

        Args:
            config_path: Caminho para arquivo YAML de configuração
        """
        self.config = self.DEFAULT_CONFIG.copy()

        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                self._update_config(self.config, user_config)

        # Configurar dispositivo
        self.device = self._setup_device()

    def _update_config(self, base: Dict, update: Dict):
        """Atualiza configuração recursivamente."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_config(base[key], value)
            else:
                base[key] = value

    def _setup_device(self) -> str:
        """Configura dispositivo de processamento."""
        device_pref = self.config.get('device', 'auto')

        if device_pref == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_pref

        return device

    def get(self, key: str, default: Any = None) -> Any:
        """Obtém valor de configuração."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def print_device_info(self):
        """Imprime informações sobre o dispositivo."""
        print("\n" + "=" * 60)
        print("🎬 SISTEMA DE ANÁLISE DE VÍDEO")
        print("=" * 60)

        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🚀 GPU detectada: {gpu_name}")
            print(f"   VRAM disponível: {vram:.1f} GB")
        else:
            print("⚠️  GPU não detectada - usando CPU (mais lento)")

        print("=" * 60 + "\n")
