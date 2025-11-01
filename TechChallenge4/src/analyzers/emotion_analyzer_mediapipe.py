"""
Analisador de Emoções usando MediaPipe + FER

Estratégia:
- MediaPipe Face Detection: Detector de faces do Google (extremamente preciso)
- FER: Biblioteca moderna para reconhecimento de emoções
"""

import cv2
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import mediapipe as mp
from fer import FER


class EmotionAnalyzerMediaPipe:
    """Analisador de emoções usando MediaPipe + FER"""

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Inicializa o analisador.

        Args:
            confidence_threshold: Confiança mínima para detecção (0.0 a 1.0)
        """
        self.confidence_threshold = confidence_threshold

        # Inicializar MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 = modelo completo (melhor para faces distantes)
            min_detection_confidence=confidence_threshold
        )

        # Inicializar FER para emoções
        self.fer = FER(mtcnn=False)  # Não usar MTCNN (já temos MediaPipe)

        # Mapeamento de emoções FER -> nosso formato
        self.emotion_map = {
            'angry': 'raiva',
            'disgust': 'nojo',
            'fear': 'medo',
            'happy': 'feliz',
            'sad': 'triste',
            'surprise': 'surpreso',
            'neutral': 'neutro'
        }

        print("[INFO] EmotionAnalyzerMediaPipe configurado")
        print("   Detector: MediaPipe Face Detection")
        print("   Classificador: FER")

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detecta rostos no frame usando MediaPipe.

        Args:
            frame: Frame do vídeo (BGR)

        Returns:
            Lista de dicionários com detecções
        """
        detections = []
        h, w, _ = frame.shape

        # Converter BGR -> RGB para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar faces
        results = self.face_detection.process(frame_rgb)

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

                if face_img.shape[0] > 10 and face_img.shape[1] > 10:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'face_img': face_img
                    })

        return detections

    def classify_emotion(self, face_crop: np.ndarray) -> Dict[str, Any]:
        """
        Classifica emoção do rosto usando FER.

        Args:
            face_crop: Imagem do rosto recortada (BGR)

        Returns:
            Dicionário com emoção e scores
        """
        try:
            # FER espera RGB
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            # Analisar emoções
            emotions = self.fer.detect_emotions(face_rgb)

            if emotions and len(emotions) > 0:
                # Pegar primeira detecção
                emotion_scores = emotions[0]['emotions']

                # Encontrar emoção dominante
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                dominant_score = emotion_scores[dominant_emotion]

                # Mapear para português
                emotion_pt = self.emotion_map.get(dominant_emotion, 'neutro')

                return {
                    'emotion': emotion_pt,
                    'confidence': dominant_score,
                    'all_emotions': {
                        self.emotion_map[k]: v
                        for k, v in emotion_scores.items()
                    }
                }
            else:
                return {
                    'emotion': 'neutro',
                    'confidence': 0.0,
                    'all_emotions': {}
                }

        except Exception as e:
            return {
                'emotion': 'neutro',
                'confidence': 0.0,
                'all_emotions': {}
            }

    def process_scene(
        self,
        video_path: str,
        output_path: str,
        show_scores: bool = True
    ) -> Dict[str, Any]:
        """
        Processa uma cena completa.

        Args:
            video_path: Caminho do vídeo de entrada
            output_path: Caminho do vídeo de saída
            show_scores: Mostrar scores de confiança

        Returns:
            Dicionário com estatísticas
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

        # Configurações do vídeo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Writer de vídeo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Estatísticas
        frame_data = []
        emotion_counts = {}
        frames_with_faces = 0
        max_faces_in_frame = 0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detectar faces
            faces = self.detect_faces(frame)

            if len(faces) > 0:
                frames_with_faces += 1
                max_faces_in_frame = max(max_faces_in_frame, len(faces))

            # Processar cada face
            for face in faces:
                x1, y1, x2, y2 = face['bbox']

                # Classificar emoção
                emotion_result = self.classify_emotion(face['face_img'])
                emotion = emotion_result['emotion']
                confidence = emotion_result['confidence']

                # Atualizar contadores
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                # Desenhar bbox e label
                color = (0, 255, 0)  # Verde
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label
                label = f"{emotion}"
                if show_scores:
                    label += f" ({confidence:.2f})"

                # Fundo para o texto
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Mostrar contador de pessoas
            people_text = f"Pessoas: {len(faces)}"
            cv2.putText(frame, people_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        # Calcular emoção dominante
        if emotion_counts:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        else:
            dominant_emotion = 'neutro'

        return {
            'scene_name': Path(video_path).stem,
            'total_frames': total_frames,
            'frames_with_faces': frames_with_faces,
            'detection_rate': (frames_with_faces / total_frames * 100) if total_frames > 0 else 0,
            'total_detections': sum(emotion_counts.values()),
            'emotion_counts': emotion_counts,
            'dominant_emotion': dominant_emotion,
            'max_people_in_frame': max_faces_in_frame,
            'output_path': output_path
        }
