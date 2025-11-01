"""
Atividade padrão quando nenhuma atividade específica é detectada.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .base_activity import BaseActivity


class UnknownActivity(BaseActivity):
    """
    Classe padrão quando nenhuma atividade é detectada com confiança suficiente.

    Sempre retorna detected=False, mas coleta informações sobre a cena.
    """

    def _get_activity_name(self) -> str:
        return "Não Identificado"

    def _get_activity_icon(self) -> str:
        return "❓"

    def _get_activity_color(self) -> Tuple[int, int, int]:
        return (128, 128, 128)  # Cinza em BGR

    def detect(
        self,
        pose_keypoints: np.ndarray,
        detected_objects: List[Dict[str, Any]],
        face_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Coleta informações sobre a cena mesmo sem detectar atividade específica.

        Args:
            pose_keypoints: Array de keypoints
            detected_objects: Objetos detectados
            face_data: Dados faciais opcionais

        Returns:
            Dict com detected=False e metadados sobre a cena
        """
        evidence = ["Nenhuma atividade específica identificada"]

        # Coletar informações sobre objetos visíveis
        visible_objects = [obj['class'] for obj in detected_objects if obj['confidence'] > 0.5]

        # Verificar se pose é válida
        pose_valid = False
        if pose_keypoints is not None:
            try:
                pose_valid = self.validate_pose_keypoints(pose_keypoints)
            except:
                pass

        # Detectar movimento (se houver histórico)
        movement_detected = False

        metadata = {
            'visible_objects': visible_objects,
            'pose_valid': pose_valid,
            'movement_detected': movement_detected,
            'reason': 'no_clear_pattern'
        }

        if visible_objects:
            evidence.append(f"Objetos visíveis: {', '.join(set(visible_objects))}")

        return {
            'detected': False,
            'confidence': 0.0,
            'evidence': evidence,
            'metadata': metadata
        }
