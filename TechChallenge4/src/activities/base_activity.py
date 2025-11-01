"""
Classe base abstrata para detecção de atividades.

Define o contrato e comportamentos comuns para todos os detectores de atividades.
Implementa Template Method Pattern e fornece utilitários reutilizáveis.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import math


class BaseActivity(ABC):
    """
    Classe abstrata base para todos os detectores de atividades.

    Fornece:
    - Interface padrão que todas as atividades devem implementar
    - Métodos utilitários reutilizáveis (cálculos geométricos)
    - Lógica de negócio comum (thresholds, validação)
    - Template methods que definem o fluxo geral de detecção
    """

    def __init__(self, confidence_threshold: float = 0.6):
        """
        Inicializa a atividade base.

        Args:
            confidence_threshold: Confiança mínima para considerar atividade detectada (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.activity_name = self._get_activity_name()
        self.activity_icon = self._get_activity_icon()
        self.color = self._get_activity_color()

    # ==================== MÉTODOS ABSTRATOS (DEVEM SER IMPLEMENTADOS) ====================

    @abstractmethod
    def _get_activity_name(self) -> str:
        """
        Retorna o nome legível da atividade.

        Returns:
            Nome da atividade (ex: "Lendo", "Escrevendo")
        """
        pass

    @abstractmethod
    def _get_activity_icon(self) -> str:
        """
        Retorna o emoji/ícone representativo da atividade.

        Returns:
            Emoji da atividade (ex: "📖", "✍️")
        """
        pass

    @abstractmethod
    def _get_activity_color(self) -> Tuple[int, int, int]:
        """
        Retorna a cor BGR para visualização da atividade.

        Returns:
            Tupla BGR (ex: (0, 255, 0) para verde)
        """
        pass

    @abstractmethod
    def detect(
        self,
        pose_keypoints: np.ndarray,
        detected_objects: List[Dict[str, Any]],
        face_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Método principal de detecção da atividade.

        Args:
            pose_keypoints: Array numpy (17, 3) com keypoints YOLO pose.
                Cada linha: [x, y, confidence]
            detected_objects: Lista de objetos detectados.
                Cada dict: {'class': str, 'confidence': float, 'bbox': [x1,y1,x2,y2]}
            face_data: Dados faciais opcionais (landmarks, expressões).

        Returns:
            Dict com estrutura:
            {
                'detected': bool - Se atividade foi detectada,
                'confidence': float - Confiança da detecção (0-1),
                'evidence': List[str] - Razões que levaram à detecção,
                'metadata': Dict - Informações adicionais específicas da atividade
            }

        Raises:
            ValueError: Se pose_keypoints não tiver o shape correto.
        """
        pass

    # ==================== MÉTODOS CONCRETOS (IMPLEMENTADOS NA BASE) ====================

    def is_detected(self, detection_result: Dict[str, Any]) -> bool:
        """
        Verifica se a atividade foi detectada com confiança suficiente.

        Args:
            detection_result: Resultado do método detect()

        Returns:
            True se detectado com confiança >= threshold
        """
        return (
            detection_result.get('detected', False) and
            detection_result.get('confidence', 0.0) >= self.confidence_threshold
        )

    def get_annotation_text(self, confidence: float) -> str:
        """
        Retorna texto formatado para anotação visual.

        Args:
            confidence: Confiança da detecção (0-1)

        Returns:
            Texto formatado: "📖 Lendo (85%)"
        """
        return f"{self.activity_icon} {self.activity_name} ({confidence * 100:.0f}%)"

    # ==================== UTILITÁRIOS GEOMÉTRICOS ====================

    @staticmethod
    def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calcula o ângulo formado por três pontos (p1-p2-p3).

        Útil para verificar ângulos de articulações (ex: cotovelo, joelho).

        Args:
            p1: Primeiro ponto [x, y] ou [x, y, conf]
            p2: Ponto central [x, y] ou [x, y, conf]
            p3: Terceiro ponto [x, y] ou [x, y, conf]

        Returns:
            Ângulo em graus (0-180)
        """
        # Extrair apenas coordenadas x, y
        p1 = p1[:2]
        p2 = p2[:2]
        p3 = p3[:2]

        # Vetores
        v1 = p1 - p2
        v2 = p3 - p2

        # Calcular ângulo usando produto escalar
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    @staticmethod
    def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Calcula distância euclidiana entre dois pontos.

        Args:
            p1: Primeiro ponto [x, y] ou [x, y, conf]
            p2: Segundo ponto [x, y] ou [x, y, conf]

        Returns:
            Distância em pixels
        """
        p1 = p1[:2]
        p2 = p2[:2]
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def check_keypoint_valid(keypoint: np.ndarray, min_confidence: float = 0.3) -> bool:
        """
        Verifica se um keypoint é válido (confiança suficiente).

        Args:
            keypoint: Keypoint [x, y, confidence]
            min_confidence: Confiança mínima aceitável

        Returns:
            True se keypoint é válido
        """
        if len(keypoint) < 3:
            return False
        return keypoint[2] >= min_confidence

    def check_object_presence(
        self,
        objects: List[Dict[str, Any]],
        target_classes: List[str],
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Verifica se objetos específicos estão presentes na cena.

        Args:
            objects: Lista de objetos detectados
            target_classes: Classes de objetos a procurar
            min_confidence: Confiança mínima para considerar objeto

        Returns:
            Dict com estrutura:
            {
                'found': bool - Se algum objeto foi encontrado,
                'objects': List[Dict] - Objetos encontrados,
                'confidence': float - Maior confiança encontrada
            }
        """
        found_objects = []
        max_confidence = 0.0

        for obj in objects:
            if (obj['class'] in target_classes and
                obj['confidence'] >= min_confidence):
                found_objects.append(obj)
                max_confidence = max(max_confidence, obj['confidence'])

        return {
            'found': len(found_objects) > 0,
            'objects': found_objects,
            'confidence': max_confidence
        }

    def get_keypoint_by_name(self, pose_keypoints: np.ndarray, name: str) -> Optional[np.ndarray]:
        """
        Obtém keypoint específico pelo nome (padrão YOLO pose).

        YOLO Pose Keypoints (COCO format):
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
        13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

        Args:
            pose_keypoints: Array (17, 3) com keypoints
            name: Nome do keypoint

        Returns:
            Keypoint [x, y, conf] ou None se inválido
        """
        keypoint_map = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }

        if name not in keypoint_map:
            return None

        idx = keypoint_map[name]
        if idx >= len(pose_keypoints):
            return None

        kp = pose_keypoints[idx]
        return kp if self.check_keypoint_valid(kp) else None

    def validate_pose_keypoints(self, pose_keypoints: np.ndarray) -> bool:
        """
        Valida se o array de keypoints tem o formato correto.

        Args:
            pose_keypoints: Array de keypoints

        Returns:
            True se válido

        Raises:
            ValueError: Se formato incorreto
        """
        if pose_keypoints is None:
            return False

        if not isinstance(pose_keypoints, np.ndarray):
            raise ValueError("pose_keypoints deve ser um numpy array")

        if pose_keypoints.shape != (17, 3):
            raise ValueError(f"pose_keypoints deve ter shape (17, 3), recebido: {pose_keypoints.shape}")

        return True

    def __repr__(self) -> str:
        """Representação string da atividade."""
        return f"{self.__class__.__name__}(name='{self.activity_name}', threshold={self.confidence_threshold})"
