"""
Detector de atividade de uso de celular.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .base_activity import BaseActivity


class PhoneActivity(BaseActivity):
    """Detecta uso de celular."""

    ELBOW_ANGLE_MIN = 60.0
    ELBOW_ANGLE_MAX = 120.0

    def _get_activity_name(self) -> str:
        return "Telefone"

    def _get_activity_icon(self) -> str:
        return "📱"

    def _get_activity_color(self) -> Tuple[int, int, int]:
        return (147, 20, 255)  # Rosa em BGR

    def detect(
        self,
        pose_keypoints: np.ndarray,
        detected_objects: List[Dict[str, Any]],
        face_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.validate_pose_keypoints(pose_keypoints):
            return self._create_negative_result("Pose inválida")

        evidence = []
        confidence_score = 0.0

        # CRITÉRIO OBRIGATÓRIO: Deve ter celular detectado OU mão na orelha
        phone_result = self.check_object_presence(
            detected_objects, ['cell phone', 'phone'], min_confidence=0.5
        )

        hand_near_ear = self._check_hand_near_ear(pose_keypoints)

        # Se não tem celular detectado E mão não está na orelha = não é uso de telefone
        if not phone_result['found'] and not hand_near_ear:
            return self._create_negative_result("Nenhum celular detectado e mão não está na orelha")

        # Celular detectado (50%)
        if phone_result['found']:
            confidence_score += 0.50
            evidence.append("Celular detectado")

            phone_obj = phone_result['objects'][0]

            # Verificar mão próxima ao celular (20%)
            if self._check_hand_near_object(pose_keypoints, phone_obj['bbox']):
                confidence_score += 0.20
                evidence.append("Mão próxima ao celular")

        # Mão próxima à orelha - ligação (30%)
        if hand_near_ear:
            confidence_score += 0.30
            evidence.append("Possível ligação telefônica")

        # Ângulo do braço (não usado mais - removido para simplificar)

        detected = confidence_score >= self.confidence_threshold

        metadata = {
            'phone_in_hand': phone_result['found'],
            'possible_call': self._check_hand_near_ear(pose_keypoints),
            'usage_mode': 'calling' if self._check_hand_near_ear(pose_keypoints) else 'browsing'
        }

        return {
            'detected': detected,
            'confidence': min(1.0, confidence_score),
            'evidence': evidence,
            'metadata': metadata
        }

    def _check_hand_near_object(self, keypoints: np.ndarray, bbox: List[float]) -> bool:
        """Verifica se mão está próxima ao objeto."""
        obj_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        bbox_width = bbox[2] - bbox[0]

        for wrist_idx in [9, 10]:
            wrist = keypoints[wrist_idx]
            if self.check_keypoint_valid(wrist):
                dist = self.calculate_distance(wrist[:2], obj_center)
                if dist < bbox_width * 1.5:
                    return True
        return False

    def _check_hand_near_ear(self, keypoints: np.ndarray) -> bool:
        """Verifica se mão está próxima à orelha."""
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_ear = keypoints[3]
        right_ear = keypoints[4]

        # Verificar mão direita com orelha direita
        if self.check_keypoint_valid(right_wrist) and self.check_keypoint_valid(right_ear):
            dist = self.calculate_distance(right_wrist[:2], right_ear[:2])
            if dist < 50:  # pixels
                return True

        # Verificar mão esquerda com orelha esquerda
        if self.check_keypoint_valid(left_wrist) and self.check_keypoint_valid(left_ear):
            dist = self.calculate_distance(left_wrist[:2], left_ear[:2])
            if dist < 50:
                return True

        return False

    def _check_elbow_angle(self, keypoints: np.ndarray) -> bool:
        """Verifica ângulo do cotovelo."""
        for side in [(5, 7, 9), (6, 8, 10)]:  # esquerdo, direito
            shoulder, elbow, wrist = [keypoints[i] for i in side]
            if all(self.check_keypoint_valid(kp) for kp in [shoulder, elbow, wrist]):
                angle = self.calculate_angle(shoulder, elbow, wrist)
                if self.ELBOW_ANGLE_MIN <= angle <= self.ELBOW_ANGLE_MAX:
                    return True
        return False

    def _create_negative_result(self, reason: str) -> Dict[str, Any]:
        return {'detected': False, 'confidence': 0.0, 'evidence': [reason], 'metadata': {}}
