"""
ETAPA 3 (Híbrido): Interpretação de Atividades com Abordagem Híbrida

CLI para detectar atividades usando o melhor de VideoMAE e Análise de Pose:
- VideoMAE: Dançando, Acenando, Caretas (atividades dinâmicas)
- Pose+Objetos: Trabalhando, Lendo, Telefone (atividades estáticas)
"""

import argparse
from pathlib import Path
import sys
import os
from datetime import datetime

# Forçar uso de PyTorch no transformers (desabilitar TensorFlow)
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

# Configurar encoding UTF-8 para evitar erros no Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analyzers.hybrid_analyzer import HybridAnalyzer


def generate_hybrid_report(all_results, output_dir):
    """Gera relatório consolidado em Markdown."""
    report_path = output_dir / "relatorio_hibrido.md"

    # Calcular estatísticas globais
    total_clips = sum(r['total_clips'] for r in all_results)
    all_activities = {}
    all_methods = {}

    for r in all_results:
        for activity, count in r['activity_distribution'].items():
            all_activities[activity] = all_activities.get(activity, 0) + count
        for method, count in r.get('method_distribution', {}).items():
            all_methods[method] = all_methods.get(method, 0) + count

    most_common = max(all_activities, key=all_activities.get) if all_activities else 'N/A'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 🔀 Relatório de Análise Híbrida de Atividades\n\n")
        f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Método:** Híbrido (VideoMAE + Análise de Pose)\n")
        f.write(f"**Total de cenas:** {len(all_results)}\n")
        f.write(f"**Total de clipes:** {total_clips}\n")
        f.write(f"**Atividade predominante:** {most_common}\n\n")
        f.write("---\n\n")

        # Estratégia do método híbrido
        f.write("## 🎯 Estratégia Híbrida\n\n")
        f.write("| Método | Atividades | Vantagem |\n")
        f.write("|--------|-----------|----------|\n")
        f.write("| 🤖 VideoMAE | Dançando, Acenando, Caretas, Gargalhadas | Entende movimento temporal |\n")
        f.write("| 🎯 Pose+Objetos | Trabalhando, Lendo, Telefone | Detecta objetos e poses estáticas |\n\n")

        # Estatísticas de métodos usados
        f.write("## 📊 Métodos Utilizados\n\n")
        f.write("| Método | Clipes | Percentual |\n")
        f.write("|--------|--------|------------|\n")
        for method, count in sorted(all_methods.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_clips * 100) if total_clips > 0 else 0
            icon = '🤖' if method == 'VideoMAE' else '🎯' if method == 'Pose' else '❓'
            f.write(f"| {icon} {method} | {count} | {percentage:.1f}% |\n")
        f.write("\n")

        # Distribuição de atividades
        f.write("## 🎬 Distribuição Geral de Atividades\n\n")
        emoji_map = {
            'Trabalhando': '💻',
            'Lendo': '📖',
            'Telefone': '📱',
            'Dançando': '💃',
            'Acenando': '👋',
            'Fazendo Caretas': '😜',
            'Dando Gargalhadas': '😂',
            'Não Identificado': '❓'
        }

        f.write("| Atividade | Ocorrências | Percentual |\n")
        f.write("|-----------|-------------|------------|\n")
        for activity, count in sorted(all_activities.items(), key=lambda x: x[1], reverse=True):
            emoji = emoji_map.get(activity, '❓')
            percentage = (count / total_clips * 100) if total_clips > 0 else 0
            f.write(f"| {emoji} {activity} | {count} | {percentage:.1f}% |\n")
        f.write("\n")

        # Detalhes por cena
        f.write("## 🎥 Análise por Cena\n\n")
        for idx, result in enumerate(all_results, 1):
            scene_name = Path(result['scene_path']).stem
            f.write(f"### {scene_name}\n\n")
            f.write(f"- **Atividade dominante:** {result['dominant_activity']}\n")
            f.write(f"- **Total de clipes:** {result['total_clips']}\n")
            f.write(f"- **FPS:** {result['fps']}\n")
            f.write(f"- **Frames totais:** {result['total_frames']}\n\n")

            # Distribuição de atividades nesta cena
            if result.get('activity_distribution'):
                f.write("**Atividades detectadas:**\n\n")
                for activity, count in result['activity_distribution'].items():
                    emoji = emoji_map.get(activity, '❓')
                    percentage = (count / result['total_clips'] * 100) if result['total_clips'] > 0 else 0
                    f.write(f"- {emoji} {activity}: {count} ({percentage:.1f}%)\n")
                f.write("\n")

            # Métodos usados nesta cena
            if result.get('method_distribution'):
                f.write("**Métodos utilizados:**\n\n")
                for method, count in result['method_distribution'].items():
                    percentage = (count / result['total_clips'] * 100) if result['total_clips'] > 0 else 0
                    icon = '🤖' if method == 'VideoMAE' else '🎯' if method == 'Pose' else '❓'
                    f.write(f"- {icon} {method}: {count} ({percentage:.1f}%)\n")
                f.write("\n")

            f.write("---\n\n")

    print(f"\n📄 Relatório gerado: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description='Etapa 3 (Híbrido): Interpretação de Atividades',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input-dir', '-i', default='output/cenas', help='Diretório com cenas')
    parser.add_argument('--output-dir', '-o', default='output/hibrido', help='Diretório de saída')
    parser.add_argument('--device', '-d', default='auto', choices=['auto', 'cuda', 'cpu'], help='Dispositivo')
    parser.add_argument('--clip-duration', '-c', type=float, default=2.0, help='Duração do clipe em segundos')
    parser.add_argument('--overlap', type=float, default=1.0, help='Overlap entre clipes em segundos')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🔀 ETAPA 3 (HÍBRIDO): INTERPRETAÇÃO DE ATIVIDADES")
    print("="*60 + "\n")

    print("📋 Estratégia Híbrida:")
    print("   🤖 VideoMAE: Dançando, Acenando, Caretas, Gargalhadas")
    print("   🎯 Pose+Objetos: Trabalhando, Lendo, Telefone\n")

    # Criar analisador híbrido
    analyzer = HybridAnalyzer(device=args.device)

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
        output_path = output_dir / f"{scene_name}_hibrido.mp4"

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

        # Mostrar distribuição de atividades
        if result.get('activity_distribution'):
            print(f"   📈 Distribuição de atividades:")
            emoji_map = {
                'Trabalhando': '💻',
                'Lendo': '📖',
                'Telefone': '📱',
                'Dançando': '💃',
                'Acenando': '👋',
                'Fazendo Caretas': '😜',
                'Dando Gargalhadas': '😂',
                'Não Identificado': '❓'
            }
            for activity, count in result['activity_distribution'].items():
                emoji = emoji_map.get(activity, '❓')
                percentage = (count / result['total_clips'] * 100) if result['total_clips'] > 0 else 0
                print(f"      • {emoji} {activity}: {count} clipes ({percentage:.1f}%)")

        # Mostrar distribuição de métodos
        if result.get('method_distribution'):
            print(f"   🔧 Métodos utilizados:")
            for method, count in result['method_distribution'].items():
                percentage = (count / result['total_clips'] * 100) if result['total_clips'] > 0 else 0
                icon = '🤖' if method == 'VideoMAE' else '🎯' if method == 'Pose' else '❓'
                print(f"      • {icon} {method}: {count} clipes ({percentage:.1f}%)")

        print(f"   📄 Vídeo anotado: {output_path}\n")

    # Estatísticas gerais
    total_clips = sum(r['total_clips'] for r in all_results)
    all_activities = {}
    for r in all_results:
        for activity, count in r['activity_distribution'].items():
            all_activities[activity] = all_activities.get(activity, 0) + count

    all_methods = {}
    for r in all_results:
        for method, count in r.get('method_distribution', {}).items():
            all_methods[method] = all_methods.get(method, 0) + count

    most_common = max(all_activities, key=all_activities.get) if all_activities else 'N/A'

    print("\n" + "="*60)
    print("✅ ETAPA 3 (HÍBRIDO) CONCLUÍDA COM SUCESSO!")
    print(f"📊 Total de clipes analisados: {total_clips}")
    print(f"🎯 Atividade mais comum: {most_common}")

    if all_methods:
        print(f"\n🔧 Métodos mais utilizados:")
        for method, count in sorted(all_methods.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_clips * 100) if total_clips > 0 else 0
            icon = '🤖' if method == 'VideoMAE' else '🎯' if method == 'Pose' else '❓'
            print(f"   • {icon} {method}: {count} clipes ({percentage:.1f}%)")

    # Gerar relatório em Markdown
    generate_hybrid_report(all_results, output_dir)

    print("="*60 + "\n")


if __name__ == '__main__':
    main()
