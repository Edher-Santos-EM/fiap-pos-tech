"""Sistema de logging."""

import logging
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_dir: str = "output/relatorios", level=logging.INFO) -> logging.Logger:
    """
    Configura logger para o sistema.

    Args:
        name: Nome do logger
        log_dir: Diretório para arquivos de log
        level: Nível de logging

    Returns:
        Logger configurado
    """
    # Criar diretório de logs
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Nome do arquivo de log com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name}_{timestamp}.log"

    # Configurar logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Handler para arquivo
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)

    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Adicionar handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
