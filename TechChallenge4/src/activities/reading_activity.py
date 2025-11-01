"""
Detector de atividade de leitura.

Identifica quando uma pessoa está lendo baseado em:
- Inclinação da cabeça para baixo
- Presença de objetos de leitura (livro, jornal, tablet, etc.)
- Posição das mãos
- Pessoa relativamente estática
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .base_activity import BaseActivity


class ReadingActivity(BaseActivity):
    """
    Detecta atividade de leitura.

    Critérios de detecção:
    1. Cabeça inclinada para baixo (ângulo 20-70°)
    2. Presença de objetos de leitura: book, newspaper, magazine, paper, tablet, laptop
    3. Mãos em posição apropriada: abaixo dos ombros, próximas ao objeto
    4. Pessoa relativamente estática

    Pesos de confiança:
    - Inclinação da cabeça: 35%
    - Objeto de leitura detectado: 40%
    - Posição das mãos: 25%
    """

    # Objetos que indicam leitura
    # Nota: laptop foi removido (agora indica "Trabalhando")
    READING_OBJECTS = [
        'book', 'newspaper', 'magazine', 'paper',
        'tablet', 'notebook'
    ]

    # Faixas de ângulo de inclinação da cabeça
    HEAD_TILT_MIN = 20.0  # graus
    HEAD_TILT_MAX = 70.0  # graus

    def _get_activity_name(self) -> str:
        return "Lendo"

    def _get_activity_icon(self) -> str:
        return "📖"

    def _get_activity_color(self) -> Tuple[int, int, int]:
        return (0, 255, 0)  # Verde em BGR

    def detect(
        self,
        pose_keypoints: np.ndarray,
        detected_objects: List[Dict[str, Any]],
        face_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detecta se a pessoa está lendo.

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

        # CRITÉRIO OBRIGATÓRIO: Deve ter objeto de leitura
        obj_result = self.check_object_presence(
            detected_objects,
            self.READING_OBJECTS,
            min_confidence=0.5
        )

        if not obj_result['found']:
            return self._create_negative_result("Nenhum objeto de leitura detectado (livro, jornal, tablet, etc.)")

        # Se chegou aqui, tem objeto de leitura (50%)
        confidence_score += 0.50
        obj_names = [obj['class'] for obj in obj_result['objects']]
        evidence.append(f"Objeto de leitura: {', '.join(obj_names)}")

        # 1. Verificar inclinação da cabeça (30%)
        head_angle = self._check_head_tilt(pose_keypoints)
        if head_angle is not None:
            if self.HEAD_TILT_MIN <= head_angle <= self.HEAD_TILT_MAX:
                confidence_score += 0.30
                evidence.append(f"Cabeça inclinada ({head_angle:.1f}°)")

        # 2. Verificar posição das mãos (20%)
        if self._check_hands_position(pose_keypoints):
            confidence_score += 0.20
            evidence.append("Mãos em posição de leitura")

        # Determinar se detectado
        detected = confidence_score >= self.confidence_threshold

        # Preparar metadados
        metadata = {
            'head_angle': head_angle,
            'objects_found': [obj['class'] for obj in obj_result.get('objects', [])],
            'hands_stable': self._check_hands_position(pose_keypoints)
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
        quanto a cabeça está inclinada para baixo.

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
        Verifica se mãos estão em posição apropriada para leitura.

        Critérios:
        - Mãos abaixo dos ombros (Y maior que ombros)
        - Pelo menos uma mão em posição correta

        Args:
            keypoints: Array de keypoints

        Returns:
            True se mãos em posição apropriada
        """
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]

        # Verificar pelo menos uma combinação mão-ombro válida
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
