"""
Detector de atividade de careta (grimacing).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .base_activity import BaseActivity


class GrimacingActivity(BaseActivity):
    """Detecta caretas através de análise facial (requer face_data)."""

    def _get_activity_name(self) -> str:
        return "Fazendo Careta"

    def _get_activity_icon(self) -> str:
        return "😜"

    def _get_activity_color(self) -> Tuple[int, int, int]:
        return (0, 165, 255)  # Laranja em BGR

    def detect(
        self,
        pose_keypoints: np.ndarray,
        detected_objects: List[Dict[str, Any]],
        face_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detecta caretas. Requer face_data com landmarks faciais.

        Se face_data não disponível, usa heurística baseada em pose.
        """
        evidence = []
        confidence_score = 0.0

        # Se temos dados faciais, usar análise avançada
        if face_data and 'landmarks' in face_data:
            result = self._analyze_facial_landmarks(face_data)
            confidence_score = result['confidence']
            evidence = result['evidence']
        else:
            # Heurística simples: detectar mãos próximas ao rosto
            if self.validate_pose_keypoints(pose_keypoints):
                if self._check_hands_near_face(pose_keypoints):
                    confidence_score = 0.5
                    evidence.append("Mãos próximas ao rosto")

        detected = confidence_score >= self.confidence_threshold

        metadata = {
            'facial_analysis_available': face_data is not None,
            'grimace_type': 'unknown'
        }

        return {
            'detected': detected,
            'confidence': confidence_score,
            'evidence': evidence,
            'metadata': metadata
        }

    def _analyze_facial_landmarks(self, face_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa landmarks faciais para detectar caretas.

        Esta é uma implementação simplificada. Para produção,
        seria necessário análise mais sofisticada de landmarks.
        """
        evidence = []
        score = 0.0

        # Análise simplificada
        if 'emotion' in face_data:
            emotion = face_data['emotion']
            # Emoções exageradas podem indicar careta
            if emotion in ['surprise', 'fear', 'disgust']:
                score += 0.3
                evidence.append(f"Expressão exagerada: {emotion}")

        if 'mouth_open' in face_data and face_data['mouth_open']:
            score += 0.4
            evidence.append("Boca exageradamente aberta")

        if 'eyebrows_raised' in face_data and face_data['eyebrows_raised']:
            score += 0.3
            evidence.append("Sobrancelhas levantadas")

        return {'confidence': min(1.0, score), 'evidence': evidence}

    def _check_hands_near_face(self, keypoints: np.ndarray) -> bool:
        """Verifica se mãos estão próximas ao rosto."""
        nose = keypoints[0]
        if not self.check_keypoint_valid(nose):
            return False

        for wrist_idx in [9, 10]:
            wrist = keypoints[wrist_idx]
            if self.check_keypoint_valid(wrist):
                dist = self.calculate_distance(nose[:2], wrist[:2])
                if dist < 100:  # pixels
                    return True

        return False

    def _create_negative_result(self, reason: str) -> Dict[str, Any]:
        return {'detected': False, 'confidence': 0.0, 'evidence': [reason], 'metadata': {}}
