"""
Detector de atividade de aceno/tchau.

Identifica quando uma pessoa está acenando (dando tchau) baseado em:
- Mão levantada acima dos ombros
- Braço estendido ou semi-estendido
- Posição característica de aceno
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .base_activity import BaseActivity


class WavingActivity(BaseActivity):
    """
    Detecta atividade de acenar/dar tchau.

    Critérios de detecção (ACENO DEVE SER CLARO):
    1. APENAS UMA mão levantada (não ambas como na dança)
    2. Mão BEM acima do ombro (pelo menos 50px)
    3. Mão no nível da cabeça ou acima
    4. Mão afastada lateralmente do corpo

    Pesos de confiança:
    - Mão bem acima do ombro: 50% (25% se apenas ligeiramente acima)
    - Mão no nível da cabeça: 40% (20% se apenas próxima)
    - Mão afastada lateralmente: 10%
    - Rejeita se ambas as mãos estão elevadas (possível dança)
    """

    def _get_activity_name(self) -> str:
        return "Acenando"

    def _get_activity_icon(self) -> str:
        return "👋"

    def _get_activity_color(self) -> Tuple[int, int, int]:
        return (255, 200, 0)  # Amarelo em BGR

    def detect(
        self,
        pose_keypoints: np.ndarray,
        detected_objects: List[Dict[str, Any]],
        face_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detecta se a pessoa está acenando.

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

        # Verificar ambos os braços
        left_waving = self._check_waving_hand(pose_keypoints, 'left')
        right_waving = self._check_waving_hand(pose_keypoints, 'right')

        # ACENO DEVE SER CLARO: Apenas UMA mão levantada (não ambas como na dança)
        # Se ambas as mãos estão muito elevadas, provavelmente é dança, não aceno
        if left_waving['score'] >= 0.7 and right_waving['score'] >= 0.7:
            return self._create_negative_result("Ambas as mãos elevadas (possível dança)")

        # Usar o braço com maior confiança (deve ter score significativo)
        if left_waving['score'] > 0 or right_waving['score'] > 0:
            if left_waving['score'] >= right_waving['score']:
                confidence_score = left_waving['score']
                evidence = left_waving['evidence']
                side = 'esquerdo'
            else:
                confidence_score = right_waving['score']
                evidence = right_waving['evidence']
                side = 'direito'
        else:
            return self._create_negative_result("Nenhuma mão levantada detectada")

        # Determinar se detectado
        detected = confidence_score >= self.confidence_threshold

        # Preparar metadados
        metadata = {
            'waving_hand': side,
            'left_score': left_waving['score'],
            'right_score': right_waving['score']
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

    def _check_waving_hand(self, keypoints: np.ndarray, side: str) -> Dict[str, Any]:
        """
        Verifica se uma mão específica está em posição de aceno.

        Args:
            keypoints: Array de keypoints
            side: 'left' ou 'right'

        Returns:
            Dict com 'score' e 'evidence'
        """
        if side == 'left':
            shoulder_idx = 5
            wrist_idx = 9
        else:
            shoulder_idx = 6
            wrist_idx = 10

        shoulder = keypoints[shoulder_idx]
        wrist = keypoints[wrist_idx]
        nose = keypoints[0]

        # Verificar se keypoints são válidos
        if not (self.check_keypoint_valid(shoulder) and
                self.check_keypoint_valid(wrist) and
                self.check_keypoint_valid(nose)):
            return {'score': 0.0, 'evidence': []}

        score = 0.0
        evidence = []
        head_y = nose[1]

        # CRITÉRIO 1: Mão BEM acima do ombro (50%)
        # Deve estar claramente elevada, não apenas no nível do ombro
        if wrist[1] < shoulder[1] - 50:  # Pelo menos 50px acima do ombro
            score += 0.50
            evidence.append(f"Mão {side} bem acima do ombro")
        elif wrist[1] < shoulder[1]:  # Apenas um pouco acima
            score += 0.25  # Pontuação reduzida
            evidence.append(f"Mão {side} ligeiramente acima do ombro")

        # CRITÉRIO 2: Mão no nível da cabeça ou acima (40%)
        # Este é o critério mais importante para aceno claro
        if wrist[1] <= head_y:  # No nível da cabeça ou acima
            score += 0.40
            evidence.append(f"Mão {side} no nível da cabeça")
        elif wrist[1] <= head_y + 50:  # Próximo da cabeça
            score += 0.20  # Pontuação reduzida
            evidence.append(f"Mão {side} próxima da cabeça")

        # CRITÉRIO 3: Mão lateralmente afastada do corpo (10%)
        horizontal_distance = abs(wrist[0] - shoulder[0])
        if horizontal_distance > 80:  # Bem afastada lateralmente
            score += 0.10
            evidence.append(f"Mão {side} afastada lateralmente")

        return {'score': score, 'evidence': evidence}

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
