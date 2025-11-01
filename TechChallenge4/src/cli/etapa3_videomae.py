"""
ETAPA 3 (VideoMAE): Interpretação de Atividades usando IA

CLI para detectar e classificar atividades humanas usando VideoMAE.
Muito mais preciso que análise de pose!
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analyzers.videomae_analyzer import VideoMAEAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description='Etapa 3 (VideoMAE): Interpretação de Atividades',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input-dir', '-i', default='output/cenas', help='Diretório com cenas')
    parser.add_argument('--output-dir', '-o', default='output/videomae', help='Diretório de saída')
    parser.add_argument('--device', '-d', default='auto', choices=['auto', 'cuda', 'cpu'], help='Dispositivo')
    parser.add_argument('--clip-duration', '-c', type=float, default=2.0, help='Duração do clipe em segundos')
    parser.add_argument('--overlap', type=float, default=1.0, help='Overlap entre clipes em segundos')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🤖 ETAPA 3 (VideoMAE): INTERPRETAÇÃO DE ATIVIDADES COM IA")
    print("="*60 + "\n")

    # Criar analisador VideoMAE
    analyzer = VideoMAEAnalyzer(device=args.device)

    # Listar cenas
    input_dir = Path(args.input_dir)
    scenes = sorted(input_dir.glob("cena_*.mp4"))

    if not scenes:
        print(f"❌ Nenhuma cena encontrada em {input_dir}")
        return

    print(f"📁 {len(scenes)} cenas encontradas\n")

    # Processar cada cena
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for scene_path in scenes:
        scene_name = scene_path.stem
        output_path = output_dir / f"{scene_name}_videomae.mp4"

        print(f"🎬 Processando: {scene_name}...")

        result = analyzer.process_scene(
            str(scene_path),
            str(output_path),
            clip_duration=args.clip_duration,
            overlap=args.overlap
        )

        all_results.append(result)

        print(f"   ✅ Atividade dominante: {result['dominant_activity']}")
        print(f"   📊 Total de clipes: {result['total_clips']}")

        # Mostrar distribuição
        if result.get('activity_distribution'):
            print(f"   📈 Distribuição:")
            emoji_map = {
                'Trabalhando': '💻',
                'Lendo': '📖',
                'Telefone': '📱',
                'Dançando': '💃',
                'Acenando': '👋',
                'Fazendo Caretas': '😜',
                'Não Identificado': '❓'
            }
            for activity, count in result['activity_distribution'].items():
                emoji = emoji_map.get(activity, '❓')
                percentage = (count / result['total_clips'] * 100) if result['total_clips'] > 0 else 0
                print(f"      • {emoji} {activity}: {count} clipes ({percentage:.1f}%)")

        print(f"   📄 Vídeo anotado: {output_path}\n")

    # Estatísticas gerais
    total_clips = sum(r['total_clips'] for r in all_results)
    all_activities = {}
    for r in all_results:
        for activity, count in r['activity_distribution'].items():
            all_activities[activity] = all_activities.get(activity, 0) + count

    most_common = max(all_activities, key=all_activities.get) if all_activities else 'N/A'

    print("\n" + "="*60)
    print("✅ ETAPA 3 (VideoMAE) CONCLUÍDA COM SUCESSO!")
    print(f"📊 Total de clipes analisados: {total_clips}")
    print(f"🎯 Atividade mais comum: {most_common}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
