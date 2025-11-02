"""
Pipeline completo: Detecção de Cenas → Emoções → Atividades (VideoMAE)

Executa todas as etapas do processamento de vídeo e gera relatórios.
Utiliza dois ambientes virtuais separados para evitar conflitos:
- venv_emotions: Para análise de sentimentos (TensorFlow + DeepFace)
- venv_activities: Para análise de atividades (PyTorch + VideoMAE)
"""

import subprocess
import sys
import platform
import os
from pathlib import Path

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def get_python_path(venv_name):
    """Retorna caminho do python no venv."""
    if platform.system() == "Windows":
        return f"{venv_name}\\Scripts\\python.exe"
    else:
        return f"{venv_name}/bin/python"


def check_environments():
    """Verifica se os ambientes virtuais existem."""
    venv_scenes = Path("venv_scenes")
    venv_emotions = Path("venv_emotions")
    venv_activities = Path("venv_activities")

    missing = []
    if not venv_scenes.exists():
        missing.append("venv_scenes (Etapa 1: Cenas)")
    if not venv_emotions.exists():
        missing.append("venv_emotions (Etapa 2: Emoções)")
    if not venv_activities.exists():
        missing.append("venv_activities (Etapa 3: Atividades)")

    if missing:
        print("\n" + "="*60)
        print("⚠️  AMBIENTES VIRTUAIS NÃO ENCONTRADOS")
        print("="*60)
        print("\nEste projeto requer três ambientes virtuais:")
        print("  • venv_scenes - Para detecção de cenas (leve)")
        print("  • venv_emotions - Para análise de sentimentos")
        print("  • venv_activities - Para análise de atividades")
        print("\nAmbientes faltando:")
        for env in missing:
            print(f"  ❌ {env}")
        print("\nOpções de instalação:")
        print("  1. Criar todos os ambientes:")
        print("     python setup_dual_environments.py")
        print("\n  2. Criar apenas ambiente de cenas (rápido):")
        print("     python setup_scenes_only.py")
        print("\n" + "="*60 + "\n")
        return False

    return True


def run_command(cmd, description):
    """Executa um comando e mostra o progresso."""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n❌ Erro ao executar: {description}")
        return False

    print(f"\n✅ {description} - Concluído!")
    return True


def main():
    print("\n" + "="*60)
    print("🎬 PIPELINE COMPLETO DE ANÁLISE DE VÍDEO")
    print("="*60)

    # Verificar ambientes virtuais
    if not check_environments():
        return

    # Verificar se há vídeo de entrada
    video_input = Path("videos/video.mp4")
    if not video_input.exists():
        print(f"\n❌ Vídeo de entrada não encontrado: {video_input}")
        print("   Coloque seu vídeo em: videos/video.mp4")
        print("\n💡 Dica: A pasta 'videos/' foi criada para organizar os vídeos")
        return

    print(f"\n✅ Vídeo encontrado: {video_input}")
    print(f"   Tamanho: {video_input.stat().st_size / 1024 / 1024:.2f} MB\n")

    # Obter caminhos dos interpretadores Python
    python_scenes = get_python_path("venv_scenes")
    python_emotions = get_python_path("venv_emotions")
    python_activities = get_python_path("venv_activities")

    # Etapa 1: Detecção de Cenas (usa venv_scenes - leve)
    print("\n💡 Usando ambiente: venv_scenes (OpenCV + SceneDetect)")
    if not run_command(
        f'"{python_scenes}" src/cli/etapa1_separar_cenas.py --input "{video_input}"',
        "ETAPA 1: Separação de Cenas"
    ):
        return

    # Etapa 2: Análise de Sentimentos (usa venv_emotions)
    print("\n💡 Usando ambiente: venv_emotions (TensorFlow + DeepFace)")
    if not run_command(
        f'"{python_emotions}" src/cli/etapa2_analisar_sentimentos.py',
        "ETAPA 2: Análise de Sentimentos"
    ):
        return

    # Etapa 3: Atividades (usa venv_activities)
    print("\n" + "="*60)
    print("▶ ETAPA 3: Escolha o método de análise")
    print("="*60)
    print("\n💡 Usando ambiente: venv_activities (PyTorch + VideoMAE)")
    print("\n1. Híbrido (Recomendado - Melhor de cada método)")
    print("   • VideoMAE: Dançando, Acenando, Caretas")
    print("   • Pose+Objetos: Trabalhando, Lendo, Telefone")
    print("\n2. VideoMAE (IA - Apenas atividades dinâmicas)")
    print("3. Análise de Pose (YOLO - Apenas atividades estáticas)")

    choice = input("\nEscolha (1, 2 ou 3): ").strip()

    if choice == "1":
        if not run_command(
            f'"{python_activities}" src/cli/etapa3_hibrido.py',
            "ETAPA 3: Interpretação de Atividades (Híbrido)"
        ):
            return
    elif choice == "2":
        if not run_command(
            f'"{python_activities}" src/cli/etapa3_videomae.py',
            "ETAPA 3: Interpretação de Atividades (VideoMAE)"
        ):
            return
    else:
        if not run_command(
            f'"{python_activities}" src/cli/etapa3_interpretar_atividades.py',
            "ETAPA 3: Interpretação de Atividades (Pose)"
        ):
            return

    # Finalização
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
    print("="*60)

    print("\n📁 Resultados gerados:")
    print("   • output/cenas/ - Cenas detectadas")
    print("   • output/sentimentos/ - Análise de sentimentos")

    if choice == "1":
        print("   • output/hibrido/ - Atividades detectadas (Híbrido)")
    elif choice == "2":
        print("   • output/videomae/ - Atividades detectadas (VideoMAE)")
    else:
        print("   • output/atividades/ - Atividades detectadas (Pose)")

    print("\n📄 Relatórios:")

    if choice == "1":
        print("   • Vídeos anotados com atividades detectadas (método híbrido)")
    elif choice == "2":
        # Não há relatórios markdown para VideoMAE ainda
        print("   • Vídeos anotados com atividades detectadas")
    else:
        reports = list(Path("output/atividades").glob("*.md"))
        for report in reports:
            print(f"   • {report}")

    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()
