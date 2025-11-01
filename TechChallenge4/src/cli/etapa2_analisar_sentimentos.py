"""
ETAPA 2: Análise de Sentimentos

CLI para analisar emoções faciais nas cenas.
"""

import argparse
from pathlib import Path
import sys
import os

# Configurar encoding UTF-8 para evitar erros no Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analyzers.emotion_analyzer import EmotionAnalyzer
from src.core.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Etapa 2: Análise de Sentimentos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input-dir', '-i', default='output/cenas', help='Diretório com cenas')
    parser.add_argument('--output-dir', '-o', default='output/sentimentos', help='Diretório de saída')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confiança mínima')
    parser.add_argument('--device', '-d', default='auto', choices=['auto', 'cuda', 'cpu'], help='Dispositivo')
    parser.add_argument('--no-scores', action='store_true', help='Não mostrar scores')
    # Nota: DeepFace usa RetinaFace para detecção e seus próprios modelos de emoção (baixados automaticamente)

    args = parser.parse_args()

    print("\n" + "="*60)
    print("ETAPA 2: ANALISE DE SENTIMENTOS")
    print("="*60 + "\n")

    # Criar analisador
    analyzer = EmotionAnalyzer(
        confidence_threshold=args.confidence,
        device=args.device
    )

    # Listar cenas
    input_dir = Path(args.input_dir)
    scenes = sorted(input_dir.glob("cena_*.mp4"))

    if not scenes:
        print(f"[ERRO] Nenhuma cena encontrada em {input_dir}")
        return

    print(f"[INFO] {len(scenes)} cenas encontradas\n")

    # Processar cada cena
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for scene_path in scenes:
        scene_name = scene_path.stem
        output_path = output_dir / f"{scene_name}_sentimentos.mp4"

        print(f"[PROCESSANDO] {scene_name}...")

        result = analyzer.process_scene(
            str(scene_path),
            str(output_path),
            show_scores=not args.no_scores
        )

        all_results.append(result)
        print(f"   [OK] Emocao predominante: {result['dominant_emotion']}")
        print(f"   [INFO] Total de deteccoes: {result['total_detections']}\n")

    # Gerar relatório consolidado
    total_detections = sum(r['total_detections'] for r in all_results)

    report_path = output_dir / "relatorio_sentimentos.md"
    ReportGenerator.generate_emotion_report(
        {
            'total_detections': total_detections,
            'scenes': all_results
        },
        str(report_path)
    )

    print("\n" + "="*60)
    print("[SUCESSO] ETAPA 2 CONCLUIDA!")
    print(f"[INFO] Total de deteccoes: {total_detections}")
    print(f"[INFO] Relatorio: {report_path}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
