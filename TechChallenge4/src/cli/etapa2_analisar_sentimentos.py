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

    # Mapeamento de emoções para emojis
    emotion_emojis = {
        'feliz': '😊',
        'triste': '😢',
        'raiva': '😠',
        'surpreso': '😨',
        'neutro': '😐',
        'medo': '😰',
        'nojo': '🤢',
        'desprezo': '😒'
    }

    for idx, scene_path in enumerate(scenes, 1):
        scene_name = scene_path.stem
        output_path = output_dir / f"{scene_name}_sentimentos.mp4"

        print(f"\n🎬 [{idx}/{len(scenes)}] Processando {scene_name}...")

        result = analyzer.process_scene(
            str(scene_path),
            str(output_path),
            show_scores=not args.no_scores
        )

        all_results.append(result)

        # Obter emoji da emoção predominante
        emotion = result['dominant_emotion']
        emoji = emotion_emojis.get(emotion, '😐')

        # Estatísticas da cena
        detection_rate = result['detection_rate']
        total_det = result['total_detections']
        max_people = result['max_people']

        print(f"   {emoji} Emoção predominante: {emotion.upper()}")
        print(f"   👥 Pessoas detectadas: {total_det} detecções | Taxa: {detection_rate:.1f}%")
        print(f"   📊 Max pessoas simultâneas: {max_people}")
        print(f"   ✅ Salvo em: {output_path.name}")

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

    # Calcular estatísticas globais
    total_frames = sum(r['total_frames'] for r in all_results)
    frames_with_faces = sum(r['frames_with_faces'] for r in all_results)
    global_detection_rate = (frames_with_faces / total_frames * 100) if total_frames > 0 else 0

    # Distribuição de emoções
    emotion_dist = {}
    for result in all_results:
        for emotion, count in result['emotion_distribution'].items():
            emotion_dist[emotion] = emotion_dist.get(emotion, 0) + count

    # Emoção mais comum
    most_common_emotion = max(emotion_dist, key=emotion_dist.get) if emotion_dist else 'neutro'
    emoji = emotion_emojis.get(most_common_emotion, '😐')

    print("\n" + "="*60)
    print("✅ ETAPA 2 CONCLUÍDA COM SUCESSO!")
    print("="*60)
    print(f"\n📊 ESTATÍSTICAS GLOBAIS:")
    print(f"   • Total de cenas processadas: {len(scenes)}")
    print(f"   • Total de frames analisados: {total_frames:,}")
    print(f"   • Total de detecções: {total_detections:,}")
    print(f"   • Taxa de detecção global: {global_detection_rate:.1f}%")
    print(f"   • Emoção mais comum: {emoji} {most_common_emotion.upper()} ({emotion_dist.get(most_common_emotion, 0)} detecções)")
    print(f"\n📄 Relatório completo salvo em:")
    print(f"   {report_path}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
