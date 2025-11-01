"""
Detector de atividade de escrita.

Identifica quando uma pessoa está escrevendo baseado em:
- Mão segurando caneta/lápis
- Movimento de escrita (pequenos movimentos repetitivos)
- Superfície de escrita presente
- Postura de escrita
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import numpy as np
from .base_activity import BaseActivity


class WritingActivity(BaseActivity):
    """
    Detecta atividade de escrita.

    Critérios de detecção:
    1. Mão segurando caneta/lápis (objeto detectado próximo à mão)
    2. Movimento de escrita: pequenos movimentos repetitivos da mão
    3. Superfície de escrita: mesa, caderno, papel detectados
    4. Postura de escrita: braço apoiado, cotovelo em ângulo específico

    Pesos de confiança:
    - Instrumento de escrita detectado: 45%
    - Movimento característico: 30%
    - Superfície presente: 15%
    - Postura adequada: 10%
    """

    # Objetos de escrita
    WRITING_TOOLS = ['pen', 'pencil', 'marker', 'crayon']
    WRITING_SURFACES = ['table', 'desk', 'paper', 'notebook', 'book']

    # Parâmetros de movimento
    MOVEMENT_THRESHOLD = 5.0  # pixels
    HISTORY_LENGTH = 20  # frames

    # Ângulo do cotovelo para escrita
    ELBOW_ANGLE_MIN = 60.0  # graus
    ELBOW_ANGLE_MAX = 110.0  # graus

    def __init__(self, confidence_threshold: float = 0.6):
        super().__init__(confidence_threshold)
        # Histórico de posições da mão para análise de movimento
        self.hand_history = deque(maxlen=self.HISTORY_LENGTH)

    def _get_activity_name(self) -> str:
        return "Escrevendo"

    def _get_activity_icon(self) -> str:
        return "✍️"

    def _get_activity_color(self) -> Tuple[int, int, int]:
        return (255, 165, 0)  # Laranja em BGR

    def detect(
        self,
        pose_keypoints: np.ndarray,
        detected_objects: List[Dict[str, Any]],
        face_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detecta se a pessoa está escrevendo.

        Args:
            pose_keypoints: Array (17, 3) com keypoints YOLO pose
            detected_objects: Lista de objetos detectados
            face_data: Dados faciais opcionais (não usado)

        Returns:
            Dict com 'detected', 'confidence', 'evidence', 'metadata'
        """
        if not self.validate_pose_keypoints(pose_keypoints):
            return self._create_negative_result("Pose inválida")

        evidence = []
        confidence_score = 0.0

        # CRITÉRIO OBRIGATÓRIO: Deve ter instrumento de escrita
        tool_result = self._detect_writing_tool(detected_objects, pose_keypoints)
        if not tool_result['found']:
            return self._create_negative_result("Nenhum instrumento de escrita detectado (caneta, lápis, etc.)")

        # Se chegou aqui, tem instrumento de escrita (50%)
        confidence_score += 0.50
        evidence.append(f"Instrumento de escrita: {tool_result['tool']}")

        # 1. Analisar movimento da mão (25%)
        movement_score = self._analyze_hand_movement(pose_keypoints)
        if movement_score > 0:
            confidence_score += movement_score * 0.25
            evidence.append(f"Movimento de escrita detectado ({movement_score*100:.0f}%)")

        # 2. Verificar superfície de escrita (15%)
        surface_result = self.check_object_presence(
            detected_objects,
            self.WRITING_SURFACES,
            min_confidence=0.5
        )
        if surface_result['found']:
            confidence_score += 0.15
            evidence.append("Superfície de escrita presente")

        # 3. Verificar postura de escrita (10%)
        if self._check_writing_posture(pose_keypoints):
            confidence_score += 0.10
            evidence.append("Postura de escrita adequada")

        detected = confidence_score >= self.confidence_threshold

        metadata = {
            'writing_tool': tool_result.get('tool', None),
            'movement_detected': movement_score > 0.5,
            'surface_present': surface_result['found'],
            'posture_correct': self._check_writing_posture(pose_keypoints)
        }

        return {
            'detected': detected,
            'confidence': confidence_score,
            'evidence': evidence,
            'metadata': metadata
        }

    def _detect_writing_tool(
        self,
        objects: List[Dict[str, Any]],
        keypoints: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detecta instrumento de escrita próximo à mão.

        Args:
            objects: Objetos detectados
            keypoints: Keypoints da pose

        Returns:
            Dict com 'found' e 'tool'
        """
        tool_result = self.check_object_presence(
            objects,
            self.WRITING_TOOLS,
            min_confidence=0.4
        )

        if not tool_result['found']:
            return {'found': False, 'tool': None}

        # Verificar se ferramenta está próxima à mão
        for obj in tool_result['objects']:
            bbox = obj['bbox']  # [x1, y1, x2, y2]
            obj_center = np.array([
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2
            ])

            # Verificar distância para ambas as mãos
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]

            for wrist in [left_wrist, right_wrist]:
                if self.check_keypoint_valid(wrist):
                    dist = self.calculate_distance(wrist[:2], obj_center)
                    bbox_width = bbox[2] - bbox[0]
                    if dist < bbox_width * 1.5:  # Próximo à mão
                        return {'found': True, 'tool': obj['class']}

        return {'found': False, 'tool': None}

    def _analyze_hand_movement(self, keypoints: np.ndarray) -> float:
        """
        Analisa histórico de movimentos da mão para detectar padrão de escrita.

        Escrita caracteriza-se por pequenos movimentos repetitivos.

        Args:
            keypoints: Keypoints atuais

        Returns:
            Score de movimento (0-1)
        """
        # Obter posição da mão dominante (direita por padrão)
        right_wrist = keypoints[10]
        if not self.check_keypoint_valid(right_wrist):
            left_wrist = keypoints[9]
            if not self.check_keypoint_valid(left_wrist):
                return 0.0
            current_pos = left_wrist[:2]
        else:
            current_pos = right_wrist[:2]

        # Adicionar ao histórico
        self.hand_history.append(current_pos)

        # Precisa de histórico suficiente
        if len(self.hand_history) < 10:
            return 0.0

        # Calcular movimentos entre frames consecutivos
        movements = []
        for i in range(1, len(self.hand_history)):
            dist = np.linalg.norm(self.hand_history[i] - self.hand_history[i-1])
            movements.append(dist)

        movements = np.array(movements)

        # Escrita: movimentos pequenos mas consistentes
        avg_movement = np.mean(movements)
        std_movement = np.std(movements)

        # Score baseado em movimento na faixa ideal
        if self.MOVEMENT_THRESHOLD * 0.5 <= avg_movement <= self.MOVEMENT_THRESHOLD * 2:
            if std_movement < self.MOVEMENT_THRESHOLD:  # Consistente
                return min(1.0, avg_movement / self.MOVEMENT_THRESHOLD)

        return 0.0

    def _check_writing_posture(self, keypoints: np.ndarray) -> bool:
        """
        Verifica postura de escrita (cotovelo em ângulo específico).

        Args:
            keypoints: Keypoints da pose

        Returns:
            True se postura adequada
        """
        # Verificar braço direito primeiro
        right_shoulder = keypoints[6]
        right_elbow = keypoints[8]
        right_wrist = keypoints[10]

        if all(self.check_keypoint_valid(kp) for kp in [right_shoulder, right_elbow, right_wrist]):
            angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            if self.ELBOW_ANGLE_MIN <= angle <= self.ELBOW_ANGLE_MAX:
                return True

        # Tentar braço esquerdo
        left_shoulder = keypoints[5]
        left_elbow = keypoints[7]
        left_wrist = keypoints[9]

        if all(self.check_keypoint_valid(kp) for kp in [left_shoulder, left_elbow, left_wrist]):
            angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            if self.ELBOW_ANGLE_MIN <= angle <= self.ELBOW_ANGLE_MAX:
                return True

        return False

    def _create_negative_result(self, reason: str) -> Dict[str, Any]:
        """Cria resultado negativo."""
        return {
            'detected': False,
            'confidence': 0.0,
            'evidence': [reason],
            'metadata': {}
        }
