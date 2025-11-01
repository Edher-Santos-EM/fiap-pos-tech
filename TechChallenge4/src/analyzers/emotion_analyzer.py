"""
Analisador de sentimentos faciais usando MediaPipe + DeepFace.

ETAPA 2: Análise de Sentimentos

Tecnologias:
- MediaPipe: Detecção de rostos rápida e precisa (Google)
- DeepFace: Classificação de emoções usando modelos pré-treinados
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import warnings
import urllib.request
warnings.filterwarnings('ignore')

import mediapipe as mp

from ..utils.progress_bar import create_progress_bar


class EmotionAnalyzer:
    """
    Analisa emoções faciais em vídeos usando MediaPipe + DeepFace.

    MediaPipe: Detecta rostos com alta precisão do Google
    DeepFace: Classifica 7 emoções usando modelos pré-treinados
    """

    # Emoções reconhecidas pelo DeepFace
    EMOTIONS_LIST = ['neutro', 'feliz', 'surpreso', 'triste', 'raiva', 'nojo', 'medo']

    # Configurações visuais
    EMOTIONS = {
        'feliz': ('feliz', (0, 255, 0)),      # Verde
        'triste': ('triste', (255, 0, 0)),      # Azul
        'raiva': ('raiva', (0, 0, 255)),       # Vermelho
        'surpreso': ('surpreso', (0, 255, 255)),  # Amarelo
        'neutro': ('neutro', (200, 200, 200)),  # Cinza
        'medo': ('medo', (255, 0, 255)),      # Magenta
        'nojo': ('nojo', (0, 128, 128)),       # Verde escuro
        'desprezo': ('desprezo', (128, 0, 128))  # Roxo
    }

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        device: str = 'auto'
    ):
        """
        Inicializa o analisador de emoções.

        Args:
            confidence_threshold: Threshold de confiança para detecção de faces
            device: Dispositivo ('auto', 'cuda', 'cpu')
        """
        self.confidence_threshold = confidence_threshold
        self.device = device

        # Modelos
        self.face_detection = None
        self.emotion_net = None
        self.models_loaded = False

        print(f"[INFO] EmotionAnalyzer configurado")
        print(f"   Detector: MediaPipe Face Detection")
        print(f"   Classificador: DeepFace (Emotion Recognition)")

    def _download_emotion_model(self) -> str:
        """Baixa modelo ONNX de emoções se necessário."""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / "emotion_ferplus.onnx"

        if model_path.exists():
            print(f"[OK] Modelo já existe: {model_path}")
            return str(model_path)

        print("[INFO] Baixando modelo ONNX de emoções...")
        url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"

        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"[OK] Modelo baixado: {model_path}")
            return str(model_path)
        except Exception as e:
            print(f"[ERRO] Falha no download: {e}")
            return None

    def _load_models(self):
        """Carrega MediaPipe para detecção de faces."""
        if self.models_loaded:
            return

        print("[INFO] Inicializando modelos...")

        # MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=1,  # 0: faces próximas, 1: faces distantes (até 5m)
            min_detection_confidence=self.confidence_threshold
        )
        print("[OK] MediaPipe inicializado")
        print("[INFO] DeepFace será carregado na primeira classificação")

        self.models_loaded = True

    def _preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocessa face para ONNX FER+ model."""
        # Converter para grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Redimensionar para 64x64 (FER+ model)
        resized = cv2.resize(gray, (64, 64))

        # Normalizar
        normalized = resized.astype(np.float32) / 255.0

        # Criar blob
        blob = cv2.dnn.blobFromImage(normalized, 1.0, (64, 64), (0,), swapRB=False, crop=False)

        return blob

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detecta rostos no frame usando MediaPipe.

        Args:
            frame: Frame do vídeo (BGR)

        Returns:
            Lista de dicionários com detecções:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'face_img': np.ndarray
                }
            ]
        """
        if not self.models_loaded:
            self._load_models()

        try:
            # Converter para RGB (MediaPipe usa RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detectar faces
            results = self.face_detection.process(frame_rgb)

            detections = []
            h, w = frame.shape[:2]

            if results.detections:
                for detection in results.detections:
                    # Extrair bounding box
                    bbox = detection.location_data.relative_bounding_box
                    confidence = detection.score[0]

                    # Converter para coordenadas absolutas
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)

                    # Validar coordenadas
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(x1+1, min(x2, w))
                    y2 = max(y1+1, min(y2, h))

                    # Extrair face
                    face_img = frame[y1:y2, x1:x2]

                    # Validar tamanho mínimo (DeepFace requer pelo menos 48x48)
                    if face_img.shape[0] >= 48 and face_img.shape[1] >= 48:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'face_img': face_img
                        })

            return detections

        except Exception as e:
            # Em caso de erro, retornar lista vazia
            return []

    def classify_emotion(self, face_crop: np.ndarray) -> Dict[str, Any]:
        """
        Classifica emoção do rosto usando DeepFace.

        Args:
            face_crop: Imagem do rosto recortada (BGR)

        Returns:
            Dict com:
            {
                'emotion': str,           # Emoção dominante
                'confidence': float,      # Confiança (0-1)
                'probabilities': Dict     # Todas as probabilidades
            }
        """
        try:
            from deepface import DeepFace

            # Redimensionar face para tamanho adequado (DeepFace funciona melhor com faces maiores)
            # Tamanho mínimo recomendado: 224x224 para melhor performance
            h, w = face_crop.shape[:2]

            # Se a face for muito pequena, redimensionar para 224x224
            if h < 224 or w < 224:
                target_size = 224
                face_resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
            else:
                face_resized = face_crop

            # Usar DeepFace apenas para classificação de emoções
            # (não re-detectar face, apenas classificar)
            result = DeepFace.analyze(
                face_resized,
                actions=['emotion'],
                enforce_detection=False,  # Não re-detectar face
                silent=True
            )

            # DeepFace pode retornar lista ou dict
            if isinstance(result, list):
                result = result[0]

            # Extrair emoções
            emotion_scores = result['emotion']
            dominant_emotion_en = result['dominant_emotion']

            # Mapeamento DeepFace -> Português
            emotion_map = {
                'angry': 'raiva',
                'disgust': 'nojo',
                'fear': 'medo',
                'happy': 'feliz',
                'sad': 'triste',
                'surprise': 'surpreso',
                'neutral': 'neutro'
            }

            # Converter para português
            dominant_emotion_pt = emotion_map.get(dominant_emotion_en, 'neutro')
            confidence = emotion_scores[dominant_emotion_en] / 100.0

            # Criar dict de probabilidades em português
            probabilities = {}
            for emotion_en, score in emotion_scores.items():
                emotion_pt = emotion_map.get(emotion_en, emotion_en)
                probabilities[emotion_pt] = score / 100.0

            return {
                'emotion': dominant_emotion_pt,
                'confidence': confidence,
                'probabilities': probabilities
            }

        except Exception as e:
            # Retornar fallback em caso de erro
            return self._fallback_emotion()

    def _fallback_emotion(self) -> Dict[str, Any]:
        """Retorna emoção neutra em caso de erro."""
        return {
            'emotion': 'neutro',
            'confidence': 0.5,
            'probabilities': {
                'neutro': 0.5,
                'feliz': 0.1,
                'triste': 0.1,
                'raiva': 0.1,
                'surpreso': 0.1,
                'medo': 0.05,
                'nojo': 0.025,
                'desprezo': 0.025
            }
        }

    def process_scene(
        self,
        scene_path: str,
        output_path: str,
        show_scores: bool = True
    ) -> Dict[str, Any]:
        """
        Processa cena completa, detectando e anotando emoções.

        Args:
            scene_path: Caminho da cena
            output_path: Caminho de saída
            show_scores: Mostrar scores nas anotações

        Returns:
            Dict com estatísticas da análise
        """
        cap = cv2.VideoCapture(scene_path)

        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir: {scene_path}")

        # Propriedades do vídeo
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Criar writer
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Estatísticas
        emotion_counts = {e: 0 for e in self.EMOTIONS.keys()}
        total_detections = 0
        frames_with_faces = 0
        max_faces_in_frame = 0

        # Rastreamento por frame para análise detalhada
        frame_data = []

        pbar = create_progress_bar(
            total=total_frames,
            desc="Analisando emocoes",
            unit="frame",
            colour="blue"
        )

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Detectar faces
            faces = self.detect_faces(frame)

            if len(faces) > 0:
                frames_with_faces += 1
                max_faces_in_frame = max(max_faces_in_frame, len(faces))

            # Lista de emoções detectadas neste frame
            frame_faces = []

            # 2. Classificar emoções
            for idx, face in enumerate(faces):
                try:
                    emotion_result = self.classify_emotion(face['face_img'])

                    face['emotion'] = emotion_result['emotion']
                    face['emotion_conf'] = emotion_result['confidence']
                    face['probabilities'] = emotion_result['probabilities']

                    emotion_counts[emotion_result['emotion']] += 1
                    total_detections += 1

                    # Registrar dados desta face
                    frame_faces.append({
                        'person_id': idx + 1,
                        'emotion': emotion_result['emotion'],
                        'confidence': emotion_result['confidence'],
                        'bbox': face['bbox']
                    })

                except Exception as e:
                    # Em caso de erro, usar neutro
                    face['emotion'] = 'neutro'
                    face['emotion_conf'] = 0.5
                    face['probabilities'] = {}

                    frame_faces.append({
                        'person_id': idx + 1,
                        'emotion': 'neutro',
                        'confidence': 0.5,
                        'bbox': face['bbox']
                    })

            # Salvar dados do frame
            if frame_faces:
                frame_data.append({
                    'frame_num': frame_count,
                    'timestamp': frame_count / fps if fps > 0 else 0,
                    'num_faces': len(frame_faces),
                    'faces': frame_faces
                })

            # 3. Anotar frame
            annotated_frame = self.annotate_frame(frame, faces, show_scores)

            # 4. Salvar
            writer.write(annotated_frame)

            frame_count += 1
            pbar.update(1)

        cap.release()
        writer.release()
        pbar.close()

        # Emoção predominante
        dominant_emotion = 'neutro'
        if total_detections > 0:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)

        # Taxa de detecção
        detection_rate = (frames_with_faces / total_frames * 100) if total_frames > 0 else 0

        return {
            'scene_path': scene_path,
            'output_path': output_path,
            'total_frames': total_frames,
            'frames_with_faces': frames_with_faces,
            'detection_rate': detection_rate,
            'total_detections': total_detections,
            'max_people': max_faces_in_frame,
            'emotion_distribution': emotion_counts,
            'dominant_emotion': dominant_emotion,
            'avg_faces_per_frame': total_detections / total_frames if total_frames > 0 else 0,
            'frame_data': frame_data
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        show_scores: bool = True
    ) -> np.ndarray:
        """
        Anota frame com emoções detectadas.

        Args:
            frame: Frame original
            detections: Detecções de faces
            show_scores: Mostrar scores

        Returns:
            Frame anotado
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            emotion = det.get('emotion', 'neutro')
            conf = det.get('emotion_conf', 0.0)

            # Cor por emoção
            label_text, color = self.EMOTIONS.get(emotion, self.EMOTIONS['neutro'])

            # 1. Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 2. Label
            if show_scores:
                label = f"{emotion.capitalize()} ({conf*100:.0f}%)"
            else:
                label = f"{emotion.capitalize()}"

            # Fundo do texto
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Ajustar posição do label
            label_y = y1 - 10 if y1 > 30 else y2 + label_h + 10

            cv2.rectangle(
                annotated,
                (x1, label_y - label_h - 5),
                (x1 + label_w + 5, label_y + 5),
                color, -1
            )

            # 3. Texto
            cv2.putText(
                annotated, label, (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        # 4. Contador de pessoas
        people_count = len(detections)

        # Fundo do contador
        cv2.rectangle(annotated, (5, 5), (250, 45), (0, 0, 0), -1)

        # Texto do contador
        counter_text = f"Pessoas: {people_count}"
        cv2.putText(
            annotated,
            counter_text,
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            cv2.LINE_AA
        )

        return annotated
