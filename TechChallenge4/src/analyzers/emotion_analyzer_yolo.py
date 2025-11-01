"""
Analisador de Emoções usando YOLOv11 + HSEmotion

Estratégia:
- YOLOv11: Detector ultra-rápido e preciso de faces
- HSEmotion: Modelo moderno de reconhecimento de emoções
"""

import cv2
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from ultralytics import YOLO
from hsemotion.facial_emotions import HSEmotionRecognizer


class EmotionAnalyzerYOLO:
    """Analisador de emoções usando YOLOv11 + HSEmotion"""

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Inicializa o analisador.

        Args:
            confidence_threshold: Confiança mínima para detecção (0.0 a 1.0)
        """
        self.confidence_threshold = confidence_threshold

        # Inicializar YOLOv11 para detecção de pessoas
        # Nota: YOLOv11 não tem modelo específico de faces pré-treinado
        # Vamos usar detecção de pessoas e então extrair a região da face
        print("[INFO] Carregando YOLOv11...")
        self.yolo = YOLO('yolo11n.pt')  # nano = mais rápido

        # Inicializar HSEmotion
        print("[INFO] Carregando HSEmotion...")
        self.emotion_recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')

        # Mapeamento de emoções HSEmotion -> nosso formato
        self.emotion_map = {
            'Anger': 'raiva',
            'Disgust': 'nojo',
            'Fear': 'medo',
            'Happiness': 'feliz',
            'Sadness': 'triste',
            'Surprise': 'surpreso',
            'Neutral': 'neutro',
            'Contempt': 'desprezo'
        }

        # Cascade para detectar faces dentro das pessoas detectadas
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        print("[INFO] EmotionAnalyzerYOLO configurado")
        print("   Detector: YOLOv11 + Haar Cascade")
        print("   Classificador: HSEmotion")

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detecta rostos no frame usando YOLO + Haar Cascade.

        Args:
            frame: Frame do vídeo (BGR)

        Returns:
            Lista de dicionários com detecções
        """
        detections = []
        h, w, _ = frame.shape

        # Detectar pessoas com YOLO
        results = self.yolo(frame, classes=[0], verbose=False)  # classe 0 = pessoa

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])

                if confidence >= self.confidence_threshold:
                    # Coordenadas da pessoa
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Validar coordenadas
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(x1+1, min(x2, w))
                    y2 = max(y1+1, min(y2, h))

                    # Extrair região da pessoa
                    person_roi = frame[y1:y2, x1:x2]

                    if person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
                        # Detectar face dentro da pessoa
                        gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                        faces_in_person = self.face_cascade.detectMultiScale(
                            gray_roi,
                            scaleFactor=1.1,
                            minNeighbors=4,
                            minSize=(30, 30)
                        )

                        if len(faces_in_person) > 0:
                            # Pegar primeira face detectada
                            fx, fy, fw, fh = faces_in_person[0]

                            # Coordenadas absolutas da face
                            face_x1 = x1 + fx
                            face_y1 = y1 + fy
                            face_x2 = face_x1 + fw
                            face_y2 = face_y1 + fh

                            # Extrair face
                            face_img = frame[face_y1:face_y2, face_x1:face_x2]

                            if face_img.shape[0] > 10 and face_img.shape[1] > 10:
                                detections.append({
                                    'bbox': [face_x1, face_y1, face_x2, face_y2],
                                    'confidence': confidence,
                                    'face_img': face_img
                                })

        return detections

    def classify_emotion(self, face_crop: np.ndarray) -> Dict[str, Any]:
        """
        Classifica emoção do rosto usando HSEmotion.

        Args:
            face_crop: Imagem do rosto recortada (BGR)

        Returns:
            Dicionário com emoção e scores
        """
        try:
            # HSEmotion espera RGB
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            # Analisar emoção
            emotion, scores = self.emotion_recognizer.predict_emotions(
                face_rgb,
                logits=True
            )

            # Mapear para português
            emotion_pt = self.emotion_map.get(emotion, 'neutro')

            # Converter scores para dict
            emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
            all_emotions = {
                self.emotion_map.get(label, label): float(score)
                for label, score in zip(emotion_labels, scores[0])
            }

            return {
                'emotion': emotion_pt,
                'confidence': float(max(scores[0])),
                'all_emotions': all_emotions
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
                color = (255, 0, 0)  # Azul
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
            cv2.putText(frame, people_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

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
