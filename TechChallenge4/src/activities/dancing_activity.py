"""
Detector de atividade de dança.

Identifica quando uma pessoa está dançando baseado em:
- Movimento de ambos os braços
- Postura dinâmica do corpo
- Braços em posições variadas (não estáticas)
- Movimento corporal amplo
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .base_activity import BaseActivity


class DancingActivity(BaseActivity):
    """
    Detecta atividade de dança.

    Critérios de detecção (evita confusão com caminhada):
    1. Ambos os braços ELEVADOS (não apenas balançando baixo como ao andar)
    2. Braços acima do nível dos ombros
    3. Movimento corporal amplo (braços bem afastados)
    4. Postura dinâmica (corpo em ângulo)
    5. MOVIMENTO TEMPORAL: sobe e desce dos braços, rotação do corpo

    Diferença da caminhada:
    - Caminhada: braços baixos balançando naturalmente
    - Dança: braços ELEVADOS acima dos ombros + movimento repetitivo

    Pesos de confiança (SIMPLIFICADOS - sem análise temporal):
    - Elevação dos braços: 10-40% (graduado conforme elevação)
    - Movimento corporal amplo (braços afastados): 30%
    - Postura dinâmica (corpo em ângulo): 30%

    Total possível: até 100%
    Threshold padrão: 40% (bem permissivo)

    NOTA: Análise temporal removida pois histórico é compartilhado entre pessoas.
    """

    def __init__(self, confidence_threshold: float = 0.4):
        """
        Inicializa o detector de dança.

        Args:
            confidence_threshold: Limite de confiança para detecção (padrão 40%)
        """
        super().__init__(confidence_threshold)

    def _get_activity_name(self) -> str:
        return "Dançando"

    def _get_activity_icon(self) -> str:
        return "💃"

    def _get_activity_color(self) -> Tuple[int, int, int]:
        return (255, 105, 180)  # Rosa pink em BGR

    def detect(
        self,
        pose_keypoints: np.ndarray,
        detected_objects: List[Dict[str, Any]],
        face_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detecta se a pessoa está dançando.

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

        # VERIFICAR SE PESSOA ESTÁ DEITADA - se sim, NÃO é dança
        if self._is_person_lying_down(pose_keypoints):
            return self._create_negative_result("Pessoa está deitada (não é dança)")

        # Verificar se keypoints essenciais estão presentes
        if not self._has_valid_keypoints(pose_keypoints):
            return self._create_negative_result("Keypoints essenciais ausentes")

        # Coletar evidências e calcular confiança
        evidence = []
        confidence_score = 0.0

        # CRITÉRIO 1: Elevação dos braços (10-40% graduado)
        arms_status = self._check_arms_elevation(pose_keypoints)
        if arms_status['score'] > 0:
            confidence_score += arms_status['score']
            evidence.extend(arms_status['evidence'])

        # CRITÉRIO 2: Movimento corporal amplo - braços afastados do corpo (30%)
        wide_movement = self._check_wide_arm_movement(pose_keypoints)
        if wide_movement:
            confidence_score += 0.30
            evidence.append("Braços afastados do corpo")

        # CRITÉRIO 3: Postura dinâmica (30%)
        dynamic_pose = self._check_dynamic_pose(pose_keypoints)
        if dynamic_pose:
            confidence_score += 0.30
            evidence.append("Postura dinâmica")

        # Determinar se detectado
        detected = confidence_score >= self.confidence_threshold

        # Preparar metadados
        metadata = {
            'both_arms_raised': both_arms['both_raised'],
            'wide_movement': wide_movement,
            'dynamic_pose': dynamic_pose
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
        - Ambos os ombros (5 e 6)
        - Ambos os punhos (9 e 10)
        - Quadris (11 e 12)

        Args:
            keypoints: Array de keypoints

        Returns:
            True se keypoints essenciais são válidos
        """
        # Ambos os ombros
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        if not (self.check_keypoint_valid(left_shoulder) and
                self.check_keypoint_valid(right_shoulder)):
            return False

        # Ambos os punhos
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        if not (self.check_keypoint_valid(left_wrist) and
                self.check_keypoint_valid(right_wrist)):
            return False

        # Pelo menos um quadril
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        if not (self.check_keypoint_valid(left_hip) or
                self.check_keypoint_valid(right_hip)):
            return False

        return True

    def _check_arms_elevation(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """
        Verifica elevação dos braços de forma graduada.

        Retorna score baseado em quão elevados estão os braços:
        - Ambos bem elevados: 30%
        - Ambos no nível dos ombros: 20%
        - Um bem elevado: 15%
        - Um no nível do ombro: 10%

        Args:
            keypoints: Array de keypoints

        Returns:
            Dict com 'score' e 'evidence'
        """
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]

        score = 0.0
        evidence = []

        # Verificar braço esquerdo
        left_very_raised = left_wrist[1] < left_shoulder[1] - 50  # Muito acima
        left_raised = left_wrist[1] < left_shoulder[1]  # Acima do ombro

        # Verificar braço direito
        right_very_raised = right_wrist[1] < right_shoulder[1] - 50
        right_raised = right_wrist[1] < right_shoulder[1]

        # Pontuar baseado na elevação (AUMENTADO para compensar falta de análise temporal)
        if left_very_raised and right_very_raised:
            score = 0.40
            evidence.append("Ambos os braços muito elevados")
        elif left_raised and right_raised:
            score = 0.30
            evidence.append("Ambos os braços elevados")
        elif left_very_raised or right_very_raised:
            score = 0.25
            evidence.append("Um braço muito elevado")
        elif left_raised or right_raised:
            score = 0.15
            evidence.append("Um braço elevado")

        return {'score': score, 'evidence': evidence}

    def _check_wide_arm_movement(self, keypoints: np.ndarray) -> bool:
        """
        Verifica se os braços estão afastados do corpo (movimento amplo).

        Args:
            keypoints: Array de keypoints

        Returns:
            True se movimento amplo detectado
        """
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]

        # Calcular largura dos ombros
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

        # Calcular distância dos punhos (largura dos braços)
        wrist_width = abs(right_wrist[0] - left_wrist[0])

        # Movimento amplo: punhos mais afastados que ombros (braços abertos)
        # REDUZIDO de 1.3x para 1.15x (mais permissivo)
        if wrist_width > shoulder_width * 1.15:  # 15% mais largo
            return True

        return False

    def _check_dynamic_pose(self, keypoints: np.ndarray) -> bool:
        """
        Verifica se a postura é dinâmica (corpo em movimento/ângulo).

        Uma postura dinâmica tem:
        - Quadris desalinhados verticalmente (corpo inclinado)
        - Ou ombros em ângulo
        - Ou braços em posições assimétricas

        Args:
            keypoints: Array de keypoints

        Returns:
            True se postura dinâmica detectada
        """
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]

        # Verificar se quadris estão desalinhados (corpo inclinado)
        if self.check_keypoint_valid(left_hip) and self.check_keypoint_valid(right_hip):
            hip_diff = abs(left_hip[1] - right_hip[1])
            if hip_diff > 20:  # REDUZIDO de 30px para 20px (mais permissivo)
                return True

        # Verificar se ombros estão em ângulo
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
        if shoulder_diff > 25:  # REDUZIDO de 40px para 25px
            return True

        # Verificar assimetria dos braços (alturas diferentes)
        # Calcular altura relativa de cada punho em relação ao seu ombro
        left_arm_height = left_shoulder[1] - left_wrist[1]
        right_arm_height = right_shoulder[1] - right_wrist[1]

        arm_asymmetry = abs(left_arm_height - right_arm_height)
        if arm_asymmetry > 60:  # REDUZIDO de 100px para 60px
            return True

        return False

    def _check_arm_vertical_movement(self) -> Dict[str, Any]:
        """
        Analisa movimento vertical dos braços ao longo do tempo.

        Dança típica tem sobe e desce repetitivo dos braços.

        Returns:
            Dict com 'has_movement' e 'variation'
        """
        if len(self.pose_history) < 3:
            return {'has_movement': False, 'variation': 0}

        # Coletar posições Y dos punhos ao longo do histórico
        left_wrist_positions = []
        right_wrist_positions = []

        for pose in self.pose_history:
            left_wrist = pose[9]
            right_wrist = pose[10]

            if self.check_keypoint_valid(left_wrist):
                left_wrist_positions.append(left_wrist[1])
            if self.check_keypoint_valid(right_wrist):
                right_wrist_positions.append(right_wrist[1])

        # Calcular variação (diferença entre max e min)
        total_variation = 0
        count = 0

        if len(left_wrist_positions) >= 3:
            left_variation = max(left_wrist_positions) - min(left_wrist_positions)
            total_variation += left_variation
            count += 1

        if len(right_wrist_positions) >= 3:
            right_variation = max(right_wrist_positions) - min(right_wrist_positions)
            total_variation += right_variation
            count += 1

        if count == 0:
            return {'has_movement': False, 'variation': 0}

        avg_variation = total_variation / count

        # Movimento significativo se variação > 40px (reduzido de 60px)
        # (sobe e desce dos braços ao dançar)
        has_movement = avg_variation > 40

        return {
            'has_movement': has_movement,
            'variation': avg_variation
        }

    def _check_body_rotation(self) -> Dict[str, Any]:
        """
        Detecta rotação do corpo ao longo do tempo.

        Rotação é detectada pela mudança de orientação dos ombros/quadris.

        Returns:
            Dict com 'has_rotation'
        """
        if len(self.pose_history) < 3:
            return {'has_rotation': False}

        # Calcular ângulo dos ombros em cada frame
        shoulder_angles = []

        for pose in self.pose_history:
            left_shoulder = pose[5]
            right_shoulder = pose[6]

            if self.check_keypoint_valid(left_shoulder) and self.check_keypoint_valid(right_shoulder):
                # Calcular largura horizontal dos ombros
                shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
                shoulder_angles.append(shoulder_width)

        if len(shoulder_angles) < 3:
            return {'has_rotation': False}

        # Calcular variação na largura dos ombros
        # (indica rotação do corpo - ombros ficam mais próximos/afastados na imagem)
        width_variation = max(shoulder_angles) - min(shoulder_angles)

        # Rotação significativa se variação > 25px (reduzido de 40px)
        has_rotation = width_variation > 25

        return {'has_rotation': has_rotation}

    def _is_person_lying_down(self, keypoints: np.ndarray) -> bool:
        """
        Verifica se a pessoa está em posição horizontal/deitada.
        Se estiver deitada, não pode estar dançando.
        """
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        # Verificar se todos os keypoints são válidos
        if not all([
            self.check_keypoint_valid(left_shoulder),
            self.check_keypoint_valid(right_shoulder),
            self.check_keypoint_valid(left_hip),
            self.check_keypoint_valid(right_hip)
        ]):
            return False

        # Calcular centro dos ombros e quadris
        shoulder_center = np.array([
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2
        ])

        hip_center = np.array([
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2
        ])

        # Calcular ângulo do corpo em relação à vertical
        diff = shoulder_center - hip_center
        angle_from_vertical = abs(np.degrees(np.arctan2(abs(diff[1]), abs(diff[0]))))

        # Se o ângulo está próximo de 90°, a pessoa está horizontal (deitada)
        MIN_HORIZONTAL_ANGLE = 45.0

        if angle_from_vertical > MIN_HORIZONTAL_ANGLE:
            return True  # Pessoa está deitada

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
