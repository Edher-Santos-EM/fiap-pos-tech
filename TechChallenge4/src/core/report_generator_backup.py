"""Geração de relatórios em Markdown."""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class ReportGenerator:
    """Gera relatórios consolidados em Markdown."""

    @staticmethod
    def generate_scene_report(scenes_data: Dict[str, Any], output_path: str):
        """Gera relatório de separação de cenas."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 📊 Relatório de Separação de Cenas\n\n")
            f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Vídeo:** {scenes_data.get('video_source', 'N/A')}\n")
            f.write(f"**Total de Cenas:** {scenes_data.get('total_scenes', 0)}\n\n")
            f.write("---\n\n")

            f.write("## 🎬 Cenas Detectadas\n\n")
            for scene in scenes_data.get('scenes', []):
                f.write(f"### Cena {scene['id']:03d}\n")
                f.write(f"- **Duração:** {scene['duration']:.2f}s\n")
                f.write(f"- **Frames:** {scene['start_frame']} - {scene['end_frame']}\n")
                f.write(f"- **Arquivo:** `{scene['filename']}`\n\n")

    @staticmethod
    def generate_emotion_report(emotions_data: Dict[str, Any], output_path: str):
        """Gera relatório de análise de sentimentos."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 😊 Relatório de Análise de Sentimentos\n\n")
            f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            f.write("## 📈 Resumo Executivo\n\n")
            f.write(f"- **Total de Detecções:** {emotions_data.get('total_detections', 0)}\n")
            f.write(f"- **Emoção Predominante:** {emotions_data.get('dominant_emotion', 'N/A')}\n\n")

    @staticmethod
    def generate_activity_report(activities_data: Dict[str, Any], output_path: str):
        """Gera relatório de interpretação de atividades."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 🎭 Relatório de Interpretação de Atividades\n\n")
            f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            f.write("## 📈 Resumo Executivo\n\n")
            f.write(f"- **Taxa de Identificação:** {activities_data.get('identification_rate', 0):.1f}%\n")
            f.write(f"- **Atividade Mais Comum:** {activities_data.get('most_common', 'N/A')}\n\n")

    @staticmethod
    def generate_consolidated_report(all_data: Dict[str, Any], output_path: str):
        """Gera relatório consolidado final."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 🎬 Relatório Completo de Análise de Vídeo\n\n")
            f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            f.write("## 🎯 Sumário Executivo\n\n")
            f.write(f"- **Vídeo Analisado:** {all_data.get('video_path', 'N/A')}\n")
            f.write(f"- **Cenas Detectadas:** {all_data.get('total_scenes', 0)}\n")
            f.write(f"- **Tempo de Processamento:** {all_data.get('processing_time', 0):.1f}s\n\n")
