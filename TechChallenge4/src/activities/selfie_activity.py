"""
Detector de atividade de selfie.

Identifica quando uma pessoa está tirando selfie baseado em:
- Presença de celular
- Braço estendido à frente (segurando celular)
- Celular próximo ao nível da cabeça
- Postura característica de selfie
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .base_activity import BaseActivity


class SelfieActivity(BaseActivity):
    """
    Detecta atividade de tirar selfie.

    Critérios de detecção:
    1. Braço estendido à frente (característico de segurar celular para selfie)
    2. Mão/punho no nível da cabeça ou acima dos ombros
    3. Braço afastado do corpo (não colado)

    Nota: Celular geralmente NÃO aparece na foto pois está sendo usado como câmera

    Pesos de confiança:
    - Braço estendido no nível correto: 70%
    - Punho próximo da cabeça/acima do ombro: 30%
    - Pelo menos um critério necessário para detecção
    """

    def _get_activity_name(self) -> str:
        return "Selfie"

    def _get_activity_icon(self) -> str:
        return "🤳"

    def _get_activity_color(self) -> Tuple[int, int, int]:
        return (255, 192, 203)  # Rosa em BGR

    def detect(
        self,
        pose_keypoints: np.ndarray,
        detected_objects: List[Dict[str, Any]],
        face_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detecta se a pessoa está tirando selfie.

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

        # CRITÉRIO PRINCIPAL: Braço estendido no nível da cabeça (70%)
        arm_extended = self._check_arm_extended(pose_keypoints)
        if arm_extended['is_extended']:
            confidence_score += 0.70
            evidence.append(f"Braço estendido ({arm_extended['side']})")

        # CRITÉRIO 2: Posição de selfie - punho próximo da cabeça (30%)
        selfie_pose = self._check_selfie_pose(pose_keypoints)
        if selfie_pose:
            confidence_score += 0.30
            evidence.append("Punho no nível da cabeça")

        # Verificar se pelo menos um critério foi atendido
        if confidence_score == 0:
            return self._create_negative_result("Nenhum critério de selfie detectado")

        # Determinar se detectado
        detected = confidence_score >= self.confidence_threshold

        # Preparar metadados
        metadata = {
            'arm_extended': arm_extended['is_extended'],
            'arm_side': arm_extended['side'],
            'selfie_pose': selfie_pose
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

    def _check_arm_extended(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """
        Verifica se algum braço está estendido RETO (característico de selfie).

        Critérios:
        - Punho próximo ou acima do ombro (Y menor ou igual)
        - Punho afastado do corpo (distância horizontal do ombro)
        - Braço deve estar alongado/reto (verificação do ângulo do cotovelo)

        Args:
            keypoints: Array de keypoints

        Returns:
            Dict com 'is_extended' e 'side'
        """
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]

        # Verificar braço esquerdo
        if (self.check_keypoint_valid(left_shoulder) and
            self.check_keypoint_valid(left_elbow) and
            self.check_keypoint_valid(left_wrist)):

            # Punho elevado (no nível do ombro ou acima)
            if left_wrist[1] <= left_shoulder[1] + 100:
                # Punho afastado do ombro (braço estendido)
                distance = abs(left_wrist[0] - left_shoulder[0])
                if distance > 80:
                    # Verificar se braço está reto (cotovelo alinhado)
                    if self._is_arm_straight(left_shoulder, left_elbow, left_wrist):
                        return {'is_extended': True, 'side': 'esquerdo'}

        # Verificar braço direito
        if (self.check_keypoint_valid(right_shoulder) and
            self.check_keypoint_valid(right_elbow) and
            self.check_keypoint_valid(right_wrist)):

            # Punho elevado (no nível do ombro ou acima)
            if right_wrist[1] <= right_shoulder[1] + 100:
                # Punho afastado do ombro (braço estendido)
                distance = abs(right_wrist[0] - right_shoulder[0])
                if distance > 80:
                    # Verificar se braço está reto (cotovelo alinhado)
                    if self._is_arm_straight(right_shoulder, right_elbow, right_wrist):
                        return {'is_extended': True, 'side': 'direito'}

        return {'is_extended': False, 'side': None}

    def _is_arm_straight(self, shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> bool:
        """
        Verifica se o braço está reto/alongado calculando o ângulo do cotovelo.

        Um braço reto tem o cotovelo alinhado entre o ombro e o punho,
        formando um ângulo próximo de 180 graus.

        Args:
            shoulder: Coordenadas do ombro
            elbow: Coordenadas do cotovelo
            wrist: Coordenadas do punho

        Returns:
            True se braço está suficientemente reto para selfie
        """
        # Vetores do ombro ao cotovelo e do cotovelo ao punho
        v1 = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
        v2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])

        # Calcular magnitudes
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)

        if mag1 == 0 or mag2 == 0:
            return False

        # Calcular ângulo usando produto escalar
        cos_angle = np.dot(v1, v2) / (mag1 * mag2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Evitar erros numéricos
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        # Braço está reto se ângulo for maior que 140 graus
        # (180 = totalmente reto, permitimos até 140 para flexibilidade)
        return angle_deg >= 140

    def _check_selfie_pose(self, keypoints: np.ndarray) -> bool:
        """
        Verifica se a pose geral é característica de selfie.

        Critérios:
        - Pelo menos um punho próximo ao nível da cabeça
        - Punho à frente do corpo

        Args:
            keypoints: Array de keypoints

        Returns:
            True se pose de selfie detectada
        """
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]

        if not self.check_keypoint_valid(nose):
            return False

        # Verificar se algum punho está próximo ao nível da cabeça
        head_y = nose[1]

        # Punho esquerdo
        if self.check_keypoint_valid(left_wrist):
            y_diff = abs(left_wrist[1] - head_y)
            if y_diff < 200:  # Aumentado para 200px da cabeça (mais tolerante)
                return True

        # Punho direito
        if self.check_keypoint_valid(right_wrist):
            y_diff = abs(right_wrist[1] - head_y)
            if y_diff < 200:  # Aumentado para 200px
                return True

        # Alternativa: verificar se punho está acima dos ombros
        if self.check_keypoint_valid(left_shoulder) and self.check_keypoint_valid(left_wrist):
            if left_wrist[1] < left_shoulder[1]:  # Punho acima do ombro
                return True

        if self.check_keypoint_valid(right_shoulder) and self.check_keypoint_valid(right_wrist):
            if right_wrist[1] < right_shoulder[1]:  # Punho acima do ombro
                return True

        return False

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
