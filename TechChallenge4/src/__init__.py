"""
Sistema de Análise de Vídeo Orientado a Objetos

Módulo principal com todas as funcionalidades do sistema.
"""

__version__ = "1.0.0"
__author__ = "FIAP POSE3"

from .activities import (
    BaseActivity,
    ReadingActivity,
    PhoneActivity,
    WorkingActivity,
    DancingActivity,
    UnknownActivity
)

from .analyzers import (
    EmotionAnalyzer,
    ActivityAnalyzer
)

from .detectors import SceneDetector

from .core import (
    VideoProcessor,
    ReportGenerator
)

from .utils import (
    Config,
    setup_logger,
    create_progress_bar
)

__all__ = [
    # Activities
    'BaseActivity',
    'ReadingActivity',
    'PhoneActivity',
    'WorkingActivity',
    'DancingActivity',
    'UnknownActivity',

    # Analyzers
    'EmotionAnalyzer',
    'ActivityAnalyzer',

    # Detectors
    'SceneDetector',

    # Core
    'VideoProcessor',
    'ReportGenerator',

    # Utils
    'Config',
    'setup_logger',
    'create_progress_bar'
]
