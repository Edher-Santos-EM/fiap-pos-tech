@echo off
REM ============================================================
REM Script auxiliar para gerenciar ambientes virtuais duais
REM ============================================================

setlocal enabledelayedexpansion

:menu
cls
echo.
echo ============================================================
echo  GERENCIADOR DE AMBIENTES VIRTUAIS
echo ============================================================
echo.
echo Este projeto usa DOIS ambientes virtuais:
echo.
echo   1. venv_emotions   - Analise de Sentimentos (TensorFlow)
echo   2. venv_activities - Analise de Atividades (PyTorch)
echo.
echo ============================================================
echo  OPCOES DISPONIVEIS
echo ============================================================
echo.
echo [1] Ativar venv_emotions (para Etapa 2)
echo [2] Ativar venv_activities (para Etapa 3)
echo [3] Executar pipeline completo
echo [4] Testar ambos os ambientes
echo [5] Verificar versoes instaladas
echo [6] Reinstalar ambientes
echo [S] Sair
echo.
echo ============================================================
echo.

choice /C 123456S /M "Escolha uma opcao"

if errorlevel 7 goto :exit_script
if errorlevel 6 goto :reinstall
if errorlevel 5 goto :check_versions
if errorlevel 4 goto :test_envs
if errorlevel 3 goto :run_pipeline
if errorlevel 2 goto :activate_activities
if errorlevel 1 goto :activate_emotions

:activate_emotions
echo.
echo ============================================================
echo  ATIVANDO venv_emotions
echo ============================================================
echo.
echo Este ambiente contem:
echo   - TensorFlow + DeepFace + MediaPipe
echo   - Para analise de sentimentos (Etapa 2)
echo.
if not exist venv_emotions (
    echo [ERRO] venv_emotions nao encontrado!
    echo Execute: python setup_dual_environments.py
    pause
    goto menu
)
echo Ativando venv_emotions...
echo.
cmd /K "venv_emotions\Scripts\activate.bat"
goto menu

:activate_activities
echo.
echo ============================================================
echo  ATIVANDO venv_activities
echo ============================================================
echo.
echo Este ambiente contem:
echo   - PyTorch + Transformers + YOLO
echo   - Para analise de atividades (Etapa 3)
echo.
if not exist venv_activities (
    echo [ERRO] venv_activities nao encontrado!
    echo Execute: python setup_dual_environments.py
    pause
    goto menu
)
echo Ativando venv_activities...
echo.
cmd /K "venv_activities\Scripts\activate.bat"
goto menu

:run_pipeline
echo.
echo ============================================================
echo  EXECUTANDO PIPELINE COMPLETO
echo ============================================================
echo.
if not exist venv_emotions (
    echo [ERRO] venv_emotions nao encontrado!
    echo Execute: python setup_dual_environments.py
    pause
    goto menu
)
if not exist venv_activities (
    echo [ERRO] venv_activities nao encontrado!
    echo Execute: python setup_dual_environments.py
    pause
    goto menu
)
echo Executando pipeline (usa ambos os ambientes automaticamente)...
echo.
python run_pipeline.py
echo.
pause
goto menu

:test_envs
echo.
echo ============================================================
echo  TESTANDO AMBIENTES
echo ============================================================
echo.
echo Testando venv_emotions (TensorFlow + DeepFace)...
echo.
if exist venv_emotions (
    venv_emotions\Scripts\python.exe -c "import tensorflow as tf; import deepface; import mediapipe as mp; print(f'TensorFlow: {tf.__version__}'); print(f'DeepFace: {deepface.__version__}'); print(f'MediaPipe: {mp.__version__}'); gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs: {len(gpus)}' if gpus else 'CPU mode')"
) else (
    echo [ERRO] venv_emotions nao encontrado!
)

echo.
echo ------------------------------------------------------------
echo.
echo Testando venv_activities (PyTorch + Transformers)...
echo.
if exist venv_activities (
    venv_activities\Scripts\python.exe -c "import torch; import transformers; import ultralytics; print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}'); print(f'Ultralytics: {ultralytics.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'CPU mode')"
) else (
    echo [ERRO] venv_activities nao encontrado!
)

echo.
pause
goto menu

:check_versions
echo.
echo ============================================================
echo  VERSOES INSTALADAS
echo ============================================================
echo.
echo --- venv_emotions ---
if exist venv_emotions (
    echo Python:
    venv_emotions\Scripts\python.exe --version
    echo.
    echo Principais pacotes:
    venv_emotions\Scripts\pip.exe list | findstr /I "tensorflow deepface mediapipe opencv numpy"
) else (
    echo [ERRO] venv_emotions nao encontrado!
)

echo.
echo --- venv_activities ---
if exist venv_activities (
    echo Python:
    venv_activities\Scripts\python.exe --version
    echo.
    echo Principais pacotes:
    venv_activities\Scripts\pip.exe list | findstr /I "torch transformers ultralytics opencv numpy"
) else (
    echo [ERRO] venv_activities nao encontrado!
)

echo.
pause
goto menu

:reinstall
echo.
echo ============================================================
echo  REINSTALAR AMBIENTES
echo ============================================================
echo.
echo Isso ira remover e recriar ambos os ambientes virtuais.
echo.
set /p confirm="Deseja continuar? (S/N): "
if /I "!confirm!" NEQ "S" goto menu

echo.
echo Executando setup...
python setup_dual_environments.py
pause
goto menu

:exit_script
echo.
echo Ate logo!
exit /b 0
