"""
Detector de atividade de gargalhada (laughing).
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import numpy as np
from .base_activity import BaseActivity


class LaughingActivity(BaseActivity):
    """
    Detecta gargalhadas através de movimentos corporais E emoções faciais.

    ⚠️  EMOÇÃO "FELIZ" É OBRIGATÓRIA! ⚠️
    - Sem DeepFace = NÃO detecta gargalhada
    - Emoção ≠ "happy" = NÃO é gargalhada
    - Sem face detectada = NÃO é gargalhada

    Pesos de confiança (COM integração DeepFace OBRIGATÓRIA):
    - Emoção facial "Feliz" (20-40%) - OBRIGATÓRIO!
    - Tremor/sacudida do corpo (25%)
    - Movimento de ombros (20%)
    - Cabeça inclinada (10%)
    - Mãos no rosto/barriga (5%)

    BLOQUEIOS AUTOMÁTICOS:
    - Qualquer emoção != "happy" → BLOQUEIA (100%)
    - Sem detecção de face → BLOQUEIA (100%)

    Total possível: até 100%
    Threshold padrão: 50%
    """

    HISTORY_LENGTH = 20  # Frames para análise de movimento
    MIN_BODY_SHAKE_INTENSITY = 8.0  # pixels - tremor mínimo do corpo
    MIN_HEAD_TILT_ANGLE = 15.0  # graus - inclinação da cabeça
    MIN_SHOULDER_MOVEMENT = 10.0  # pixels - movimento dos ombros

    def __init__(self, confidence_threshold: float = 0.5):  # REDUZIDO de 0.6
        super().__init__(confidence_threshold)
        self.nose_history = deque(maxlen=self.HISTORY_LENGTH)
        self.shoulder_history = deque(maxlen=self.HISTORY_LENGTH)
        self.torso_history = deque(maxlen=self.HISTORY_LENGTH)

    def _get_activity_name(self) -> str:
        return "Dando Gargalhadas"

    def _get_activity_icon(self) -> str:
        return "😂"

    def _get_activity_color(self) -> Tuple[int, int, int]:
        return (0, 140, 255)  # Laranja em BGR

    def detect(
        self,
        pose_keypoints: np.ndarray,
        detected_objects: List[Dict[str, Any]],
        face_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.validate_pose_keypoints(pose_keypoints):
            return self._create_negative_result("Pose inválida")

        # VERIFICAR SE PESSOA ESTÁ DEITADA - se sim, NÃO é gargalhada
        if self._is_person_lying_down(pose_keypoints):
            return self._create_negative_result("Pessoa está deitada (não é gargalhada)")

        evidence = []
        confidence_score = 0.0

        # VERIFICAR EMOÇÃO FACIAL - Gargalhada EXIGE felicidade! (OBRIGATÓRIO)
        has_happy_emotion = False
        if face_data and isinstance(face_data, dict):
            emotion = face_data.get('emotion', '').lower()
            emotion_confidence = face_data.get('emotion_confidence', 0.0)

            # GARGALHADA EXIGE FELICIDADE!
            # Qualquer outra emoção = NÃO é gargalhada
            if emotion not in ['happy', 'feliz']:
                # Se não está feliz, NÃO pode estar dando gargalhada
                return self._create_negative_result(
                    f"Emoção incompatível com gargalhada: {emotion} ({emotion_confidence:.1%})"
                )

            # Se chegou aqui, está feliz!
            if emotion_confidence > 0.5:
                confidence_score += 0.40
                evidence.append(f"Emoção facial: Feliz ({emotion_confidence:.1%})")
                has_happy_emotion = True
            else:
                # Feliz com baixa confiança
                confidence_score += 0.20
                evidence.append(f"Emoção facial: Feliz ({emotion_confidence:.1%})")
                has_happy_emotion = True

        else:
            # SEM dados de emoção = NÃO pode detectar gargalhada
            # (Gargalhada exige confirmação de felicidade)
            return self._create_negative_result("Sem detecção de emoção facial (gargalhada requer felicidade)")

        nose = pose_keypoints[0]
        left_shoulder = pose_keypoints[5]
        right_shoulder = pose_keypoints[6]
        left_hip = pose_keypoints[11]
        right_hip = pose_keypoints[12]

        # Atualizar históricos
        if self.check_keypoint_valid(nose):
            self.nose_history.append(nose[:2])

        if self.check_keypoint_valid(left_shoulder) and self.check_keypoint_valid(right_shoulder):
            shoulder_center = np.array([
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2
            ])
            self.shoulder_history.append(shoulder_center)

        if self.check_keypoint_valid(left_hip) and self.check_keypoint_valid(right_hip):
            torso_center = np.array([
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            ])
            self.torso_history.append(torso_center)

        # 1. Verificar tremor/sacudida do corpo (25% - reduzido de 35%)
        if len(self.torso_history) >= 15:
            shake_intensity = self._detect_body_shake(list(self.torso_history))
            if shake_intensity >= self.MIN_BODY_SHAKE_INTENSITY:
                confidence_score += 0.25
                evidence.append(f"Tremor corporal detectado (intensidade: {shake_intensity:.1f})")

        # 2. Movimento de ombros (subindo e descendo) (20% - reduzido de 25%)
        if len(self.shoulder_history) >= 10:
            shoulder_movement = self._detect_shoulder_bounce(list(self.shoulder_history))
            if shoulder_movement['detected']:
                confidence_score += 0.20
                evidence.append(f"Movimento característico de ombros ({shoulder_movement['oscillations']} oscilações)")

        # 3. Cabeça inclinada para trás ou para o lado (10% - reduzido de 20%)
        head_tilt = self._check_head_tilt(pose_keypoints)
        if head_tilt['is_tilted']:
            confidence_score += 0.10
            evidence.append(f"Cabeça inclinada (ângulo: {head_tilt['angle']:.1f}°)")

        # 4. Mãos próximas ao rosto/barriga (5% - reduzido de 10%)
        hands_position = self._check_hands_position(pose_keypoints)
        if hands_position['near_face_or_belly']:
            confidence_score += 0.05
            evidence.append(f"Mãos em posição típica de gargalhada ({hands_position['description']})")

        # 5. Postura aberta/braços não cruzados (REMOVIDO - não é essencial)
        # Total agora: Emoção (40%) + Corpo (25%) + Ombros (20%) + Cabeça (10%) + Mãos (5%) = 100%

        detected = confidence_score >= self.confidence_threshold

        return {
            'detected': detected,
            'confidence': min(1.0, confidence_score),
            'evidence': evidence,
            'metadata': {
                'shake_intensity': shake_intensity if len(self.torso_history) >= 15 else 0,
                'head_tilt': head_tilt
            }
        }

    def _detect_body_shake(self, positions: List[np.ndarray]) -> float:
        """
        Detecta tremor/sacudida do corpo típico de gargalhadas.
        Gargalhadas causam pequenas oscilações rápidas no corpo.
        """
        if len(positions) < 5:
            return 0.0

        # Calcular variação vertical (gargalhadas causam movimento vertical)
        y_positions = [pos[1] for pos in positions]

        # Calcular desvio padrão dos movimentos verticais
        y_diff = [abs(y_positions[i] - y_positions[i-1]) for i in range(1, len(y_positions))]

        if len(y_diff) == 0:
            return 0.0

        # Intensidade baseada na média dos movimentos
        intensity = np.mean(y_diff)

        # Verificar se há oscilações (não apenas movimento unidirecional)
        direction_changes = 0
        for i in range(1, len(y_diff)):
            if i > 0:
                diff_prev = y_positions[i] - y_positions[i-1]
                diff_curr = y_positions[i+1] - y_positions[i] if i+1 < len(y_positions) else 0
                if diff_prev * diff_curr < 0:
                    direction_changes += 1

        # Gargalhadas têm oscilações, não movimento linear
        if direction_changes >= 2:
            return intensity * (1 + direction_changes * 0.2)  # Bônus por oscilações

        return intensity * 0.5  # Penalidade se não há oscilações

    def _detect_shoulder_bounce(self, positions: List[np.ndarray]) -> Dict[str, Any]:
        """Detecta movimento de ombros subindo e descendo (típico de gargalhadas)."""
        result = {'detected': False, 'oscillations': 0}

        if len(positions) < 8:
            return result

        y_positions = [pos[1] for pos in positions]

        # Detectar oscilações verticais
        oscillation_count = 0
        MIN_SHOULDER_MOVE = 3.0  # pixels mínimos para contar como movimento

        for i in range(1, len(y_positions) - 1):
            diff_prev = y_positions[i] - y_positions[i-1]
            diff_next = y_positions[i+1] - y_positions[i]

            # Pico ou vale
            if abs(diff_prev) > MIN_SHOULDER_MOVE and diff_prev * diff_next < 0:
                oscillation_count += 1

        # Pelo menos 2-3 oscilações para ser considerado "bounce"
        if oscillation_count >= 2:
            # Verificar amplitude total
            amplitude = max(y_positions) - min(y_positions)
            if amplitude >= self.MIN_SHOULDER_MOVEMENT:
                result['detected'] = True
                result['oscillations'] = oscillation_count
                result['amplitude'] = amplitude

        return result

    def _check_head_tilt(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """Verifica inclinação da cabeça (para trás ou para o lado)."""
        result = {'is_tilted': False, 'angle': 0.0, 'direction': 'none'}

        nose = keypoints[0]
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]

        if not all([
            self.check_keypoint_valid(nose),
            self.check_keypoint_valid(left_shoulder),
            self.check_keypoint_valid(right_shoulder)
        ]):
            return result

        # Calcular centro dos ombros
        shoulder_center = np.array([
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2
        ])

        # Verificar inclinação para trás (nariz acima da linha dos ombros)
        nose_shoulder_diff_y = shoulder_center[1] - nose[1]

        # Calcular ângulo de inclinação lateral dos olhos
        if self.check_keypoint_valid(left_eye) and self.check_keypoint_valid(right_eye):
            eye_angle = abs(np.degrees(np.arctan2(
                left_eye[1] - right_eye[1],
                left_eye[0] - right_eye[0]
            )))

            # Cabeça inclinada lateralmente
            if eye_angle > self.MIN_HEAD_TILT_ANGLE:
                result['is_tilted'] = True
                result['angle'] = eye_angle
                result['direction'] = 'lateral'
                return result

        # Cabeça inclinada para trás (nose muito acima ou abaixo dos ombros)
        if nose_shoulder_diff_y > 120:  # pixels - cabeça para trás
            result['is_tilted'] = True
            result['angle'] = nose_shoulder_diff_y / 10  # Normalizar
            result['direction'] = 'backward'

        return result

    def _check_hands_position(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """Verifica se as mãos estão em posições típicas de gargalhada."""
        result = {'near_face_or_belly': False, 'description': ''}

        nose = keypoints[0]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        # Calcular centro da barriga/torso
        if self.check_keypoint_valid(left_hip) and self.check_keypoint_valid(right_hip):
            belly_center = np.array([
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2 - 80  # Um pouco acima dos quadris
            ])

            # Verificar mão esquerda
            if self.check_keypoint_valid(left_wrist):
                # Próxima ao rosto
                if self.check_keypoint_valid(nose):
                    dist_to_face = self.calculate_distance(left_wrist, nose)
                    if dist_to_face < 100:  # pixels
                        result['near_face_or_belly'] = True
                        result['description'] = 'mão esquerda no rosto'
                        return result

                # Próxima à barriga
                dist_to_belly = self.calculate_distance(left_wrist, belly_center)
                if dist_to_belly < 80:
                    result['near_face_or_belly'] = True
                    result['description'] = 'mão esquerda na barriga'
                    return result

            # Verificar mão direita
            if self.check_keypoint_valid(right_wrist):
                # Próxima ao rosto
                if self.check_keypoint_valid(nose):
                    dist_to_face = self.calculate_distance(right_wrist, nose)
                    if dist_to_face < 100:
                        result['near_face_or_belly'] = True
                        result['description'] = 'mão direita no rosto'
                        return result

                # Próxima à barriga
                dist_to_belly = self.calculate_distance(right_wrist, belly_center)
                if dist_to_belly < 80:
                    result['near_face_or_belly'] = True
                    result['description'] = 'mão direita na barriga'
                    return result

        return result

    def _check_open_posture(self, keypoints: np.ndarray) -> bool:
        """Verifica se a postura é aberta (braços não cruzados)."""
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]

        if not all([
            self.check_keypoint_valid(left_wrist),
            self.check_keypoint_valid(right_wrist),
            self.check_keypoint_valid(left_shoulder),
            self.check_keypoint_valid(right_shoulder)
        ]):
            return False

        # Verificar se braços não estão cruzados
        # (pulso esquerdo não deve estar do lado direito do corpo e vice-versa)
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2

        left_wrist_side = 'left' if left_wrist[0] < shoulder_center_x else 'right'
        right_wrist_side = 'left' if right_wrist[0] < shoulder_center_x else 'right'

        # Se cada pulso está do seu próprio lado, não está cruzado
        if left_wrist_side == 'left' and right_wrist_side == 'right':
            return True

        return False

    def _is_person_lying_down(self, keypoints: np.ndarray) -> bool:
        """
        Verifica se a pessoa está em posição horizontal/deitada.
        Se estiver deitada, não pode estar dando gargalhadas (em pé).
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
        # Usar mesmo threshold do SleepingActivity: 45 graus
        MIN_HORIZONTAL_ANGLE = 45.0

        if angle_from_vertical > MIN_HORIZONTAL_ANGLE:
            return True  # Pessoa está deitada

        return False

    def _create_negative_result(self, reason: str) -> Dict[str, Any]:
        return {'detected': False, 'confidence': 0.0, 'evidence': [reason], 'metadata': {}}
