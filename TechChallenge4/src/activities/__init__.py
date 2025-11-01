"""
Módulo de detecção de atividades.

Contém a classe base abstrata e todas as implementações específicas
de detecção de atividades humanas.
"""

from .base_activity import BaseActivity
from .reading_activity import ReadingActivity
from .phone_activity import PhoneActivity
from .working_activity import WorkingActivity
from .dancing_activity import DancingActivity
from .laughing_activity import LaughingActivity
from .unknown_activity import UnknownActivity

__all__ = [
    'BaseActivity',
    'ReadingActivity',
    'PhoneActivity',
    'WorkingActivity',
    'DancingActivity',
    'LaughingActivity',
    'UnknownActivity'
]
