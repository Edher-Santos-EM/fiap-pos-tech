"""
ETAPA 1: Separação de Cenas

CLI para detectar e separar cenas de um vídeo.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.detectors.scene_detector import SceneDetector
from src.core.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Etapa 1: Separação de Cenas',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', '-i', default='videos/video.mp4', help='Caminho do vídeo de entrada')
    parser.add_argument('--output-dir', '-o', default='output/cenas', help='Diretório de saída')
    parser.add_argument('--threshold', '-t', type=float, default=25.0, help='Threshold de detecção')
    parser.add_argument('--min-duration', '-d', type=float, default=1.0, help='Duração mínima da cena (segundos)')
    parser.add_argument('--codec', '-c', default='mp4v', help='Codec de vídeo')
    parser.add_argument('--no-thumbnails', action='store_true', help='Não gerar thumbnails')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🎬 ETAPA 1: SEPARAÇÃO DE CENAS")
    print("="*60 + "\n")

    # Criar detector
    detector = SceneDetector(
        video_path=args.input,
        output_dir=args.output_dir,
        threshold=args.threshold,
        min_scene_duration=args.min_duration,
        codec=args.codec
    )

    # Detectar cenas
    print("🔍 Detectando cenas...")
    scenes = detector.detect_scenes()
    print(f"\n✅ {len(scenes)} cenas detectadas")

    # Salvar cenas
    print("\n💾 Salvando cenas...")
    paths = detector.save_scenes(scenes)
    print(f"✅ {len(paths)} cenas salvas")

    # Gerar thumbnails
    if not args.no_thumbnails:
        print("\n🖼️  Gerando thumbnails...")
        thumbs = detector.generate_thumbnails(scenes)
        print(f"✅ {len(thumbs)} thumbnails gerados")

    # Exportar metadados
    print("\n📄 Exportando metadados...")
    detector.export_metadata(scenes)

    # Gerar relatório
    report_path = Path(args.output_dir) / "relatorio_cenas.md"
    ReportGenerator.generate_scene_report(
        {
            'video_source': args.input,
            'total_scenes': len(scenes),
            'scenes': scenes
        },
        str(report_path)
    )
    print(f"✅ Relatório salvo em: {report_path}")

    print("\n" + "="*60)
    print("✅ ETAPA 1 CONCLUÍDA COM SUCESSO!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
