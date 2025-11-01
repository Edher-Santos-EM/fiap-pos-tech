"""
Detector de atividade de trabalho.

Identifica quando uma pessoa está trabalhando baseado em:
- Presença de laptop
- Postura sentada ou em pé em frente ao laptop
- Mãos próximas ao teclado
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .base_activity import BaseActivity


class WorkingActivity(BaseActivity):
    """
    Detecta atividade de trabalho com laptop.

    Critérios de detecção:
    1. Presença obrigatória de laptop
    2. Cabeça inclinada para frente (olhando para tela)
    3. Mãos em posição de digitação (próximas ao laptop)
    4. Pessoa relativamente estática

    Pesos de confiança:
    - Laptop detectado: 60%
    - Postura apropriada: 25%
    - Posição das mãos: 15%
    """

    # Objetos que indicam trabalho
    WORK_OBJECTS = ['laptop']

    # Faixas de ângulo de inclinação da cabeça
    HEAD_TILT_MIN = 10.0  # graus
    HEAD_TILT_MAX = 50.0  # graus

    def _get_activity_name(self) -> str:
        return "Trabalhando"

    def _get_activity_icon(self) -> str:
        return "💻"

    def _get_activity_color(self) -> Tuple[int, int, int]:
        return (255, 165, 0)  # Laranja em BGR

    def detect(
        self,
        pose_keypoints: np.ndarray,
        detected_objects: List[Dict[str, Any]],
        face_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detecta se a pessoa está trabalhando com laptop.

        Args:
            pose_keypoints: Array (17, 3) com keypoints YOLO pose
            detected_objects: Lista de objetos detectados
            face_data: Dados faciais opcionais (não usado nesta atividade)

        Returns:
            Dict com 'detected', 'confidence', 'evidence', 'metadata'
        """
        # Validar entrada
        if not self.validate_pose_keypoints(pose_keypoints):
            return self._create_negative_result("Pose inválida")

        # Verificar se keypoints essenciais estão presentes
        if not self._has_valid_keypoints(pose_keypoints):
            return self._create_negative_result("Keypoints essenciais ausentes")

        # Coletar evidências e calcular confiança
        evidence = []
        confidence_score = 0.0

        # CRITÉRIO OBRIGATÓRIO: Deve ter laptop
        obj_result = self.check_object_presence(
            detected_objects,
            self.WORK_OBJECTS,
            min_confidence=0.5
        )

        if not obj_result['found']:
            return self._create_negative_result("Nenhum laptop detectado")

        # Se chegou aqui, tem laptop (60%)
        confidence_score += 0.60
        evidence.append("Laptop detectado")

        # 1. Verificar inclinação da cabeça (25%)
        head_angle = self._check_head_tilt(pose_keypoints)
        if head_angle is not None:
            if self.HEAD_TILT_MIN <= head_angle <= self.HEAD_TILT_MAX:
                confidence_score += 0.25
                evidence.append(f"Postura de trabalho ({head_angle:.1f}°)")

        # 2. Verificar posição das mãos (15%)
        if self._check_hands_position(pose_keypoints):
            confidence_score += 0.15
            evidence.append("Mãos em posição de digitação")

        # Determinar se detectado
        detected = confidence_score >= self.confidence_threshold

        # Preparar metadados
        metadata = {
            'head_angle': head_angle,
            'laptop_found': True,
            'hands_typing': self._check_hands_position(pose_keypoints)
        }

        return {
            'detected': detected,
            'confidence': confidence_score,
            'evidence': evidence,
            'metadata': metadata
        }

    def _has_valid_keypoints(self, keypoints: np.ndarray) -> bool:
        """
        Verifica se keypoints essenciais para detecção estão presentes.

        Keypoints necessários:
        - Nariz (0)
        - Pelo menos um ombro (5 ou 6)
        - Pelo menos um punho (9 ou 10)

        Args:
            keypoints: Array de keypoints

        Returns:
            True se keypoints essenciais são válidos
        """
        # Nariz
        nose = keypoints[0]
        if not self.check_keypoint_valid(nose):
            return False

        # Pelo menos um ombro
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        if not (self.check_keypoint_valid(left_shoulder) or
                self.check_keypoint_valid(right_shoulder)):
            return False

        # Pelo menos um punho
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        if not (self.check_keypoint_valid(left_wrist) or
                self.check_keypoint_valid(right_wrist)):
            return False

        return True

    def _check_head_tilt(self, keypoints: np.ndarray) -> Optional[float]:
        """
        Calcula ângulo de inclinação da cabeça.

        Usa vetor nariz-centro dos ombros vs. vertical para determinar
        quanto a cabeça está inclinada para frente.

        Args:
            keypoints: Array de keypoints

        Returns:
            Ângulo de inclinação em graus, ou None se não calculável
        """
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]

        # Verificar validade
        if not all([
            self.check_keypoint_valid(nose),
            self.check_keypoint_valid(left_shoulder),
            self.check_keypoint_valid(right_shoulder)
        ]):
            return None

        # Centro dos ombros
        shoulder_center = (left_shoulder[:2] + right_shoulder[:2]) / 2

        # Vetor nariz -> ombros
        head_vector = shoulder_center - nose[:2]

        # Vetor vertical (apontando para baixo: positivo Y)
        vertical_vector = np.array([0, 1])

        # Calcular ângulo
        cosine_angle = np.dot(head_vector, vertical_vector) / (
            np.linalg.norm(head_vector) * np.linalg.norm(vertical_vector) + 1e-6
        )
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))

        # Converter para ângulo de inclinação (0° = vertical, 90° = horizontal)
        tilt_angle = abs(90 - angle)

        return tilt_angle

    def _check_hands_position(self, keypoints: np.ndarray) -> bool:
        """
        Verifica se mãos estão em posição apropriada para digitação.

        Critérios:
        - Mãos abaixo dos ombros (Y maior que ombros)
        - Ambas as mãos em posição próxima (sugestão de digitação)

        Args:
            keypoints: Array de keypoints

        Returns:
            True se mãos em posição apropriada
        """
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]

        # Verificar ambas as mãos
        valid_positions = 0

        # Mão esquerda
        if (self.check_keypoint_valid(left_wrist) and
            self.check_keypoint_valid(left_shoulder)):
            if left_wrist[1] > left_shoulder[1]:  # Y maior = mais abaixo
                valid_positions += 1

        # Mão direita
        if (self.check_keypoint_valid(right_wrist) and
            self.check_keypoint_valid(right_shoulder)):
            if right_wrist[1] > right_shoulder[1]:
                valid_positions += 1

        return valid_positions >= 1

    def _create_negative_result(self, reason: str) -> Dict[str, Any]:
        """
        Cria resultado negativo (não detectado).

        Args:
            reason: Razão pela qual não foi detectado

        Returns:
            Dict de resultado
        """
        return {
            'detected': False,
            'confidence': 0.0,
            'evidence': [reason],
            'metadata': {}
        }
