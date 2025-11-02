#!/usr/bin/env python3
"""
Script Rápido: Criar apenas ambiente para Etapa 1 (Cenas)
Este é um ambiente leve que só precisa de OpenCV e SceneDetect
"""

import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, description=None):
    """Executa comando."""
    if description:
        print(f"  {description}")

    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def get_python_cmd():
    """Encontra Python 3.11."""
    cmds = ["py -3.11", "python3.11", "python3", "python"]

    for cmd in cmds:
        try:
            result = subprocess.run(
                f'{cmd} --version',
                shell=True,
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0 and "3.11" in result.stdout:
                return cmd
        except:
            continue

    return "python"


def main():
    print("\n" + "="*60)
    print("  CRIANDO AMBIENTE PARA ETAPA 1 (Cenas)")
    print("="*60 + "\n")

    # Verificar se estamos no diretório correto
    if not Path("run_pipeline.py").exists():
        print("[ERRO] Execute a partir do diretório raiz do projeto")
        sys.exit(1)

    python_cmd = get_python_cmd()
    print(f"[1/4] Python: {python_cmd}")

    # Criar venv_scenes
    print("\n[2/4] Criando venv_scenes...")
    if Path("venv_scenes").exists():
        print("  Ambiente já existe, pulando...")
    else:
        if not run_command(f'{python_cmd} -m venv venv_scenes', "Criando ambiente"):
            print("[ERRO] Falha ao criar ambiente")
            sys.exit(1)
        print("  OK!")

    # Atualizar pip
    print("\n[3/4] Atualizando pip...")
    if platform.system() == "Windows":
        pip_path = "venv_scenes\\Scripts\\python.exe"
    else:
        pip_path = "venv_scenes/bin/python"

    run_command(
        f'{pip_path} -m pip install --upgrade pip setuptools wheel',
        "Atualizando pip"
    )

    # Instalar dependências
    print("\n[4/4] Instalando dependências (OpenCV + SceneDetect)...")
    print("  Isso pode demorar 2-3 minutos...\n")

    if not run_command(
        f'{pip_path} -m pip install -r requirements_scenes.txt',
        "Instalando pacotes"
    ):
        print("\n[AVISO] Alguns pacotes podem ter falhado, mas continuando...")

    print("\n" + "="*60)
    print("  AMBIENTE CRIADO COM SUCESSO!")
    print("="*60 + "\n")

    print("Para usar:")
    if platform.system() == "Windows":
        print("  venv_scenes\\Scripts\\activate")
    else:
        print("  source venv_scenes/bin/activate")

    print("\nExecutar Etapa 1:")
    print("  python src/cli/etapa1_separar_cenas.py\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelado pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERRO] {e}")
        sys.exit(1)
