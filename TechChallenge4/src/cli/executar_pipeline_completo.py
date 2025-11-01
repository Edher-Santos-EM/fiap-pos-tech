"""
ETAPA 4: Pipeline Completo

CLI para executar todas as etapas sequencialmente.
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.detectors.scene_detector import SceneDetector
from src.analyzers.emotion_analyzer import EmotionAnalyzer
from src.analyzers.activity_analyzer import ActivityAnalyzer
from src.core.report_generator import ReportGenerator
from src.utils.config import Config


class VideoPipeline:
    """Pipeline completo de análise de vídeo."""

    def __init__(self, video_path: str, config: Config):
        self.video_path = video_path
        self.config = config
        self.device = config.device

    def run_full_pipeline(self):
        """Executa pipeline completo."""
        start_time = time.time()

        print("\n" + "="*60)
        print("🎬 PIPELINE COMPLETO DE ANÁLISE DE VÍDEO")
        print("="*60 + "\n")

        self.config.print_device_info()

        # Etapa 1: Separação de Cenas
        print("\n" + "="*60)
        print("🎬 ETAPA 1: SEPARAÇÃO DE CENAS")
        print("="*60 + "\n")

        detector = SceneDetector(
            video_path=self.video_path,
            output_dir='output/cenas',
            threshold=self.config.get('scene_detection.threshold', 25.0),
            min_scene_duration=self.config.get('scene_detection.min_duration', 1.0),
            codec=self.config.get('scene_detection.codec', 'mp4v'),
            device=self.device
        )

        scenes = detector.detect_scenes()
        detector.save_scenes(scenes)
        detector.generate_thumbnails(scenes)
        detector.export_metadata(scenes)

        print(f"\n✅ Etapa 1 concluída: {len(scenes)} cenas detectadas\n")

        # Etapa 2: Análise de Sentimentos
        print("\n" + "="*60)
        print("😊 ETAPA 2: ANÁLISE DE SENTIMENTOS")
        print("="*60 + "\n")

        emotion_analyzer = EmotionAnalyzer(
            confidence_threshold=self.config.get('emotion_analysis.confidence_threshold', 0.5),
            device=self.device
        )

        scene_paths = sorted(Path('output/cenas').glob("cena_*.mp4"))
        emotion_results = []

        for scene_path in scene_paths:
            scene_name = scene_path.stem
            output_path = f"output/sentimentos/{scene_name}_sentimentos.mp4"

            result = emotion_analyzer.process_scene(str(scene_path), output_path)
            emotion_results.append(result)

        print(f"\n✅ Etapa 2 concluída: {len(emotion_results)} cenas analisadas\n")

        # Etapa 3: Interpretação de Atividades
        print("\n" + "="*60)
        print("🏃 ETAPA 3: INTERPRETAÇÃO DE ATIVIDADES")
        print("="*60 + "\n")

        activity_analyzer = ActivityAnalyzer(
            confidence_threshold=self.config.get('activity_analysis.confidence_threshold', 0.6),
            device=self.device
        )

        activity_results = []

        for scene_path in scene_paths:
            scene_name = scene_path.stem
            output_path = f"output/atividades/{scene_name}_atividades.mp4"

            result = activity_analyzer.process_scene(str(scene_path), output_path)
            activity_results.append(result)

        print(f"\n✅ Etapa 3 concluída: {len(activity_results)} cenas analisadas\n")

        # Relatório Consolidado
        print("\n" + "="*60)
        print("📊 GERANDO RELATÓRIO CONSOLIDADO")
        print("="*60 + "\n")

        end_time = time.time()
        processing_time = end_time - start_time

        report_data = {
            'video_path': self.video_path,
            'total_scenes': len(scenes),
            'processing_time': processing_time,
            'device': self.device
        }

        report_path = 'output/relatorios/relatorio_completo.md'
        ReportGenerator.generate_consolidated_report(report_data, report_path)

        print(f"✅ Relatório consolidado: {report_path}")

        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETO CONCLUÍDO!")
        print(f"⏱️  Tempo total: {processing_time:.1f}s")
        print(f"🖥️  Dispositivo usado: {self.device.upper()}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline Completo de Análise de Vídeo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', '-i', required=True, help='Vídeo de entrada')
    parser.add_argument('--config', '-c', help='Arquivo de configuração YAML')
    parser.add_argument('--device', '-d', default='auto', choices=['auto', 'cuda', 'cpu'])

    args = parser.parse_args()

    # Carregar configuração
    config = Config(args.config)

    # Sobrescrever device se especificado
    if args.device != 'auto':
        config.config['device'] = args.device
        config.device = config._setup_device()

    # Criar e executar pipeline
    pipeline = VideoPipeline(args.input, config)
    pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()
