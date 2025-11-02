#!/usr/bin/env python3
"""
Script de Configuração de Ambientes Duais
==========================================
Cria dois ambientes virtuais separados:
- venv_emotions: Análise de Sentimentos (DeepFace + TensorFlow)
- venv_activities: Análise de Atividades (VideoMAE + PyTorch)

Compatível com Windows, Linux e macOS
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path


def print_header(text):
    """Imprime cabeçalho formatado."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(step, total, text):
    """Imprime passo da instalação."""
    print(f"[{step}/{total}] {text}...")


def run_command(cmd, description=None, check=True, capture=False):
    """Executa comando e trata erros."""
    if description:
        print(f"    - {description}")

    try:
        if capture:
            result = subprocess.run(
                cmd,
                shell=True,
                check=check,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        else:
            result = subprocess.run(cmd, shell=True, check=check)
            return result.returncode == 0
    except subprocess.CalledProcessError as e:
        if check:
            print(f"    [ERRO] Comando falhou: {cmd}")
            print(f"    {e}")
            return False
        return False
    except Exception as e:
        print(f"    [ERRO] {e}")
        return False


def find_python311():
    """Encontra Python 3.11 no sistema."""
    print_step(1, 8, "Verificando Python 3.11")

    python_commands = [
        "python3.11",
        "python3",
        "python",
    ]

    if platform.system() == "Windows":
        python_commands.insert(0, "py -3.11")

    for cmd in python_commands:
        try:
            version_output = run_command(
                f'{cmd} --version',
                capture=True,
                check=False
            )

            if version_output and "3.11" in version_output:
                print(f"    OK: {cmd}")
                print(f"    Versão: {version_output}")
                return cmd
        except:
            continue

    print("\n    [ERRO] Python 3.11 não encontrado!")
    print("\n    Por favor, instale Python 3.11:")
    print("      https://www.python.org/downloads/")
    print("\n    Ou se já instalado, adicione ao PATH do sistema.\n")
    sys.exit(1)


def detect_gpu():
    """Detecta GPU NVIDIA e versão CUDA disponível."""
    print_step(2, 8, "Detectando GPU NVIDIA e CUDA")

    gpu_info = {
        'has_gpu': False,
        'gpu_name': None,
        'cuda_version': None,
        'driver_version': None
    }

    try:
        # Detectar GPU
        gpu_name = run_command(
            "nvidia-smi --query-gpu=name --format=csv,noheader",
            capture=True,
            check=False
        )

        if gpu_name:
            gpu_info['has_gpu'] = True
            gpu_info['gpu_name'] = gpu_name.strip()

            # Detectar versão do driver
            driver_version = run_command(
                "nvidia-smi --query-gpu=driver_version --format=csv,noheader",
                capture=True,
                check=False
            )
            if driver_version:
                gpu_info['driver_version'] = driver_version.strip()

            # Detectar versão CUDA suportada
            cuda_output = run_command(
                "nvidia-smi",
                capture=True,
                check=False
            )

            if cuda_output and "CUDA Version:" in cuda_output:
                # Extrair versão CUDA da saída do nvidia-smi
                for line in cuda_output.split('\n'):
                    if "CUDA Version:" in line:
                        cuda_version = line.split("CUDA Version:")[-1].strip().split()[0]
                        gpu_info['cuda_version'] = cuda_version
                        break

            print(f"    OK: GPU NVIDIA detectada")
            print(f"    Modelo: {gpu_info['gpu_name']}")
            print(f"    Driver: {gpu_info['driver_version']}")
            if gpu_info['cuda_version']:
                print(f"    CUDA suportado: {gpu_info['cuda_version']}")

                # Determinar qual versão CUDA instalar
                cuda_major = int(gpu_info['cuda_version'].split('.')[0])
                if cuda_major >= 12:
                    print(f"    → Instalando pacotes para CUDA 12.x")
                elif cuda_major == 11:
                    print(f"    → Instalando pacotes para CUDA 11.x")
                else:
                    print(f"    AVISO: Versão CUDA {gpu_info['cuda_version']} pode não ser totalmente compatível")

            return gpu_info
    except Exception as e:
        print(f"    [DEBUG] Erro ao detectar GPU: {e}")

    print("    INFO: GPU NVIDIA não detectada - instalando versão CPU")
    return gpu_info


def remove_old_venv(venv_name):
    """Remove ambiente virtual antigo se existir."""
    venv_path = Path(venv_name)

    if venv_path.exists():
        print(f"    AVISO: Ambiente '{venv_name}' já existe")
        response = input(f"    Deseja remover e recriar? (s/N): ").strip().lower()

        if response in ['s', 'y', 'sim', 'yes']:
            print(f"    Removendo {venv_name}...")
            shutil.rmtree(venv_path)
            return True
        else:
            print(f"    Mantendo {venv_name} existente...")
            return False

    return True


def create_venv(python_cmd, venv_name):
    """Cria novo ambiente virtual."""
    print(f"    Criando {venv_name}...")

    success = run_command(
        f'{python_cmd} -m venv {venv_name}',
        f"Criando {venv_name}"
    )

    if success:
        print(f"    OK: {venv_name} criado")
        return True
    else:
        print(f"    [ERRO] Falha ao criar {venv_name}")
        sys.exit(1)


def get_pip_path(venv_name):
    """Retorna caminho do pip no venv."""
    if platform.system() == "Windows":
        return f"{venv_name}\\Scripts\\pip.exe"
    else:
        return f"{venv_name}/bin/pip"


def get_python_path(venv_name):
    """Retorna caminho do python no venv."""
    if platform.system() == "Windows":
        return f"{venv_name}\\Scripts\\python.exe"
    else:
        return f"{venv_name}/bin/python"


def upgrade_pip(venv_name):
    """Atualiza pip, setuptools e wheel."""
    print(f"    Atualizando pip em {venv_name}...")

    python_path = get_python_path(venv_name)

    # Usar python -m pip para evitar problemas de lock
    success = run_command(
        f'{python_path} -m pip install --upgrade pip setuptools wheel',
        "Atualizando pip, setuptools e wheel",
        check=False  # Não falhar se houver avisos
    )

    if success or success is None:
        print("    OK: pip atualizado")
        return True
    else:
        print("    [AVISO] Pip pode não ter sido atualizado, mas continuando...")
        return True  # Continuar mesmo com aviso


def install_emotions_dependencies(venv_name, gpu_info):
    """Instala dependências para análise de emoções."""
    print(f"\n    📦 Instalando dependências de EMOÇÕES em {venv_name}...")
    print("    (DeepFace, MediaPipe, TensorFlow)\n")

    pip_path = get_pip_path(venv_name)

    # Core packages
    print("    - Instalando pacotes core...")
    run_command(
        f'{pip_path} install opencv-python numpy pillow tqdm rich '
        f'matplotlib seaborn pyyaml python-dateutil',
        check=False
    )

    # TensorFlow - GPU ou CPU
    if gpu_info['has_gpu']:
        print("\n    - Instalando TensorFlow com suporte GPU...")
        print(f"      GPU: {gpu_info['gpu_name']}")
        print("      (Isso pode demorar vários minutos...)")

        # TensorFlow 2.20+ já vem com suporte CUDA integrado
        run_command(f'{pip_path} install tensorflow>=2.20.0', check=False)

        # Instalar bibliotecas CUDA apropriadas
        cuda_version = gpu_info.get('cuda_version', '')
        if cuda_version and cuda_version.startswith('12'):
            print("\n    - Instalando bibliotecas CUDA 12.x...")
            run_command(
                f'{pip_path} install nvidia-cudnn-cu12 nvidia-cublas-cu12 '
                f'nvidia-cuda-runtime-cu12',
                check=False
            )
        elif cuda_version and cuda_version.startswith('11'):
            print("\n    - Instalando bibliotecas CUDA 11.x...")
            run_command(
                f'{pip_path} install nvidia-cudnn-cu11 nvidia-cublas-cu11',
                check=False
            )
        else:
            print("\n    - Instalando bibliotecas CUDA padrão (12.x)...")
            run_command(
                f'{pip_path} install nvidia-cudnn-cu12 nvidia-cublas-cu12',
                check=False
            )
    else:
        print("\n    - Instalando TensorFlow (CPU)...")
        run_command(f'{pip_path} install tensorflow>=2.20.0', check=False)

    # DeepFace e MediaPipe
    print("\n    - Instalando DeepFace, tf-keras e MediaPipe...")
    run_command(
        f'{pip_path} install deepface>=0.0.79 tf-keras>=2.20.0 mediapipe>=0.10.0',
        check=False
    )

    print("\n    ✅ Dependências de EMOÇÕES instaladas!")


def install_activities_dependencies(venv_name, gpu_info):
    """Instala dependências para análise de atividades."""
    print(f"\n    📦 Instalando dependências de ATIVIDADES em {venv_name}...")
    print("    (YOLO, VideoMAE, Transformers, PyTorch)\n")

    pip_path = get_pip_path(venv_name)

    # Core packages
    print("    - Instalando pacotes core...")
    run_command(
        f'{pip_path} install opencv-python numpy pillow tqdm rich '
        f'matplotlib seaborn pyyaml python-dateutil',
        check=False
    )

    # PyTorch - GPU ou CPU
    if gpu_info['has_gpu']:
        cuda_version = gpu_info.get('cuda_version', '')
        cuda_major = int(cuda_version.split('.')[0]) if cuda_version else 12

        print(f"\n    - Instalando PyTorch com suporte GPU...")
        print(f"      GPU: {gpu_info['gpu_name']}")
        print(f"      CUDA: {cuda_version}")
        print("      (Isso pode demorar vários minutos...)")

        # Escolher versão do PyTorch baseado no CUDA
        if cuda_major >= 12:
            print("      → Usando PyTorch para CUDA 12.1")
            run_command(
                f'{pip_path} install torch>=2.0.0 torchvision>=0.15.0 '
                f'--index-url https://download.pytorch.org/whl/cu121',
                check=False
            )
        elif cuda_major == 11:
            print("      → Usando PyTorch para CUDA 11.8")
            run_command(
                f'{pip_path} install torch>=2.0.0 torchvision>=0.15.0 '
                f'--index-url https://download.pytorch.org/whl/cu118',
                check=False
            )
        else:
            print(f"      ⚠️  CUDA {cuda_version} - instalando versão padrão")
            run_command(
                f'{pip_path} install torch>=2.0.0 torchvision>=0.15.0',
                check=False
            )
    else:
        print("\n    - Instalando PyTorch (CPU)...")
        run_command(
            f'{pip_path} install torch>=2.0.0 torchvision>=0.15.0',
            check=False
        )

    # YOLO
    print("\n    - Instalando Ultralytics (YOLO)...")
    run_command(f'{pip_path} install ultralytics>=8.0.0', check=False)

    # Transformers + VideoMAE
    print("\n    - Instalando Transformers e Protobuf (VideoMAE)...")
    run_command(
        f'{pip_path} install transformers>=4.57.0 protobuf>=6.33.0',
        check=False
    )

    print("\n    ✅ Dependências de ATIVIDADES instaladas!")


def test_environments():
    """Testa ambos os ambientes."""
    print_header("TESTANDO AMBIENTES")

    # Testar ambiente de emoções
    print("🔍 Testando venv_emotions (TensorFlow + DeepFace)...\n")
    python_emotions = get_python_path("venv_emotions")

    test_code_emotions = """
import sys
try:
    import tensorflow as tf
    import deepface
    import mediapipe as mp
    print(f"  ✅ TensorFlow: {tf.__version__}")
    print(f"  ✅ DeepFace: {deepface.__version__}")
    print(f"  ✅ MediaPipe: {mp.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  🎮 GPUs TensorFlow: {len(gpus)}")
    else:
        print("  💻 Modo CPU (TensorFlow)")
except Exception as e:
    print(f"  ❌ Erro: {e}")
    sys.exit(1)
"""

    run_command(f'{python_emotions} -c "{test_code_emotions}"', check=False)

    print("\n" + "-" * 70 + "\n")

    # Testar ambiente de atividades
    print("🔍 Testando venv_activities (PyTorch + Transformers)...\n")
    python_activities = get_python_path("venv_activities")

    test_code_activities = """
import sys
try:
    import torch
    import transformers
    import ultralytics
    print(f"  ✅ PyTorch: {torch.__version__}")
    print(f"  ✅ Transformers: {transformers.__version__}")
    print(f"  ✅ Ultralytics: {ultralytics.__version__}")
    if torch.cuda.is_available():
        print(f"  🎮 GPUs PyTorch: {torch.cuda.device_count()}")
    else:
        print("  💻 Modo CPU (PyTorch)")
except Exception as e:
    print(f"  ❌ Erro: {e}")
    sys.exit(1)
"""

    run_command(f'{python_activities} -c "{test_code_activities}"', check=False)


def print_next_steps():
    """Mostra próximos passos."""
    print_header("PRÓXIMOS PASSOS")

    if platform.system() == "Windows":
        activate_emotions = "venv_emotions\\Scripts\\activate"
        activate_activities = "venv_activities\\Scripts\\activate"
    else:
        activate_emotions = "source venv_emotions/bin/activate"
        activate_activities = "source venv_activities/bin/activate"

    print("✨ Dois ambientes virtuais foram criados:\n")

    print("1️⃣  venv_emotions - Para análise de SENTIMENTOS (Etapa 2)")
    print(f"   Ativar: {activate_emotions}")
    print(f"   Executar: python src/cli/etapa2_analisar_sentimentos.py\n")

    print("2️⃣  venv_activities - Para análise de ATIVIDADES (Etapa 3)")
    print(f"   Ativar: {activate_activities}")
    print(f"   Executar: python src/cli/etapa3_hibrido.py\n")

    print("🚀 Para executar o pipeline completo:")
    print("   python run_pipeline.py")
    print("   (O script ativará os ambientes automaticamente)\n")

    print("📝 Arquivos de dependências criados:")
    print("   • requirements_emotions.txt - Dependências de emoções")
    print("   • requirements_activities.txt - Dependências de atividades")

    print("\n" + "=" * 70 + "\n")


def main():
    """Função principal."""
    print_header("CONFIGURAÇÃO DE AMBIENTES DUAIS")
    print("Este script criará DOIS ambientes virtuais separados:")
    print("  • venv_emotions: TensorFlow + DeepFace (Análise de Sentimentos)")
    print("  • venv_activities: PyTorch + Transformers (Análise de Atividades)")
    print("\nIsso evita conflitos de dependências entre os frameworks.\n")

    # Verificar se há argumento --auto para pular confirmação
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        print("Modo automático: Continuando sem confirmação...")
    else:
        try:
            response = input("Deseja continuar? (S/n): ").strip().lower()
            if response and response not in ['s', 'y', 'sim', 'yes', '']:
                print("\nInstalação cancelada.")
                sys.exit(0)
        except (EOFError, KeyboardInterrupt):
            print("\nModo não-interativo detectado, continuando automaticamente...")
            pass

    # Verificar se estamos no diretório correto
    if not Path("run_pipeline.py").exists():
        print("[ERRO] Execute este script a partir do diretório raiz do projeto")
        sys.exit(1)

    # Executar etapas
    python_cmd = find_python311()
    gpu_info = detect_gpu()

    # Ambiente 1: venv_emotions
    print_step(3, 8, "Configurando ambiente venv_emotions")
    should_create = remove_old_venv("venv_emotions")
    if should_create:
        create_venv(python_cmd, "venv_emotions")

    print_step(4, 8, "Instalando dependências em venv_emotions")
    upgrade_pip("venv_emotions")
    install_emotions_dependencies("venv_emotions", gpu_info)

    # Ambiente 2: venv_activities
    print_step(5, 8, "Configurando ambiente venv_activities")
    should_create = remove_old_venv("venv_activities")
    if should_create:
        create_venv(python_cmd, "venv_activities")

    print_step(6, 8, "Instalando dependências em venv_activities")
    upgrade_pip("venv_activities")
    install_activities_dependencies("venv_activities", gpu_info)

    print_header("INSTALAÇÃO CONCLUÍDA!")

    print_step(7, 8, "Testando ambientes")
    test_environments()

    print_step(8, 8, "Finalizando")
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Instalação cancelada pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERRO] Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
