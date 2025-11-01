"""
Analisador de sentimentos faciais usando DeepFace.

ETAPA 2: Análise de Sentimentos

Tecnologias:
- DeepFace (RetinaFace): Detecção de rostos de alta precisão
- DeepFace (VGG-Face): Classificação de emoções de última geração
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..utils.progress_bar import create_progress_bar


class EmotionAnalyzer:
    """
    Analisa emoções faciais em vídeos usando DeepFace.

    DeepFace (RetinaFace): Detecta rostos com alta precisão, ignorando outros objetos
    DeepFace (VGG-Face): Classifica 7 emoções usando modelos de deep learning
    """

    # Mapeamento de emoções DeepFace para nosso sistema
    DEEPFACE_TO_SYSTEM = {
        'happy': 'feliz',
        'sad': 'triste',
        'angry': 'raiva',
        'surprise': 'surpreso',
        'neutral': 'neutro',
        'fear': 'medo',
        'disgust': 'nojo'
    }

    # Configurações visuais
    EMOTIONS = {
        'feliz': ('😊', (0, 255, 0)),      # Verde
        'triste': ('😢', (255, 0, 0)),      # Azul
        'raiva': ('😠', (0, 0, 255)),       # Vermelho
        'surpreso': ('😨', (0, 255, 255)),  # Amarelo
        'neutro': ('😐', (200, 200, 200)),  # Cinza
        'medo': ('😰', (255, 0, 255)),      # Magenta
        'nojo': ('🤢', (0, 128, 128))       # Verde escuro
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
        # Configurar dispositivo
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.confidence_threshold = confidence_threshold

        # Modelos serão carregados sob demanda
        self.deepface_loaded = False

        print(f"🚀 EmotionAnalyzer configurado para: {self.device.upper()}")
        print(f"   Detector: DeepFace (RetinaFace)")
        print(f"   Classificador: DeepFace (Emotion Model)")

    def _load_models(self):
        """Inicializa DeepFace para detecção e classificação."""

        # ═══════════════════════════════════════════════
        # INICIALIZAR DEEPFACE
        # ═══════════════════════════════════════════════
        if not self.deepface_loaded:
            try:
                from deepface import DeepFace

                print("📦 Inicializando DeepFace...")

                # Pré-carregar modelos (detector + emoções)
                # DeepFace baixa modelos automaticamente na primeira vez
                try:
                    # Fazer uma predição dummy para carregar modelos
                    dummy_img = np.zeros((48, 48, 3), dtype=np.uint8)
                    DeepFace.extract_faces(
                        img_path=dummy_img,
                        detector_backend='retinaface',
                        enforce_detection=False
                    )
                    DeepFace.analyze(
                        dummy_img,
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True
                    )
                    print("✅ DeepFace inicializado com sucesso")
                    print("   Detector: RetinaFace (alta precisão)")
                    print("   Modelo de emoções: VGG-Face")
                except:
                    print("⚠️  DeepFace será inicializado na primeira análise")

                self.deepface_loaded = True

            except ImportError:
                print("❌ DeepFace não instalado. Execute: pip install deepface")
                self.deepface_loaded = False
            except Exception as e:
                print(f"⚠️  Aviso ao inicializar DeepFace: {e}")
                self.deepface_loaded = True  # Continuar mesmo assim

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detecta rostos no frame usando DeepFace (RetinaFace/MTCNN).

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
        if not self.deepface_loaded:
            self._load_models()

        try:
            from deepface import DeepFace

            # ═══════════════════════════════════════════════
            # DETECTAR FACES COM DEEPFACE
            # ═══════════════════════════════════════════════
            # DeepFace.extract_faces usa backends otimizados para detecção facial
            # Backends disponíveis: retinaface (melhor), mtcnn, opencv, ssd
            face_objs = DeepFace.extract_faces(
                img_path=frame,
                detector_backend='retinaface',  # Mais preciso para faces
                enforce_detection=False,  # Não falhar se não detectar
                align=True  # Alinhar faces para melhor análise
            )

            detections = []
            h, w = frame.shape[:2]

            for face_obj in face_objs:
                # face_obj contém: 'face', 'facial_area', 'confidence'
                confidence = face_obj.get('confidence', 0.0)

                # Filtrar por confiança
                if confidence >= self.confidence_threshold:
                    facial_area = face_obj['facial_area']

                    # DeepFace retorna: {'x': x1, 'y': y1, 'w': width, 'h': height}
                    x1 = facial_area['x']
                    y1 = facial_area['y']
                    x2 = x1 + facial_area['w']
                    y2 = y1 + facial_area['h']

                    # Validar coordenadas
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(x1+1, min(x2, w))
                    y2 = max(y1+1, min(y2, h))

                    # Extrair face do frame original (não usar face_obj['face'] que já está processada)
                    face_img = frame[y1:y2, x1:x2]

                    # Validar tamanho mínimo
                    if face_img.shape[0] > 10 and face_img.shape[1] > 10:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'face_img': face_img
                        })

            return detections

        except ImportError:
            print("❌ DeepFace não instalado!")
            return []
        except Exception as e:
            # Em caso de erro (ex: nenhuma face detectada), retornar lista vazia
            # Isso é normal e não deve ser reportado como erro
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
                'probabilities': Dict,    # Todas as probabilidades
                'raw_deepface': Dict      # Resultado bruto do DeepFace
            }
        """
        if not self.deepface_loaded:
            self._load_models()

        try:
            from deepface import DeepFace

            # ═══════════════════════════════════════════════
            # ANÁLISE DEEPFACE
            # ═══════════════════════════════════════════════

            # DeepFace.analyze retorna análise completa
            result = DeepFace.analyze(
                face_crop,
                actions=['emotion'],          # Apenas emoções
                enforce_detection=False,      # Não re-detectar face (já temos)
                silent=True                   # Suprimir logs
            )

            # DeepFace pode retornar lista ou dict
            if isinstance(result, list):
                result = result[0]

            # ═══════════════════════════════════════════════
            # PROCESSAR RESULTADO
            # ═══════════════════════════════════════════════

            emotion_scores = result['emotion']
            # emotion_scores: {
            #     'angry': 0.01,
            #     'disgust': 0.00,
            #     'fear': 0.02,
            #     'happy': 0.85,
            #     'sad': 0.03,
            #     'surprise': 0.05,
            #     'neutral': 0.04
            # }

            # Encontrar emoção dominante
            dominant_emotion_en = result['dominant_emotion']  # Em inglês
            dominant_confidence = emotion_scores[dominant_emotion_en] / 100.0

            # Converter para português
            dominant_emotion_pt = self.DEEPFACE_TO_SYSTEM.get(
                dominant_emotion_en,
                'neutro'
            )

            # Normalizar probabilidades (DeepFace retorna 0-100)
            probabilities_pt = {}
            for emotion_en, score in emotion_scores.items():
                emotion_pt = self.DEEPFACE_TO_SYSTEM.get(emotion_en, emotion_en)
                probabilities_pt[emotion_pt] = score / 100.0

            return {
                'emotion': dominant_emotion_pt,
                'confidence': dominant_confidence,
                'probabilities': probabilities_pt,
                'raw_deepface': result
            }

        except ImportError:
            print("❌ DeepFace não instalado!")
            return self._fallback_emotion()

        except Exception as e:
            # Fallback em caso de erro
            print(f"⚠️  Erro no DeepFace: {e}")
            return self._fallback_emotion()

    def _fallback_emotion(self) -> Dict[str, Any]:
        """Retorna emoção neutra em caso de erro."""
        return {
            'emotion': 'neutro',
            'confidence': 0.5,
            'probabilities': {
                'feliz': 0.14,
                'triste': 0.14,
                'raiva': 0.14,
                'surpreso': 0.14,
                'neutro': 0.16,
                'medo': 0.14,
                'nojo': 0.14
            },
            'raw_deepface': None
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

        pbar = create_progress_bar(
            total=total_frames,
            desc=f"😊 Analisando emoções",
            unit="frame",
            colour="blue"
        )

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ═══════════════════════════════════════════════
            # PIPELINE DE ANÁLISE
            # ═══════════════════════════════════════════════

            # 1. Detectar faces (YOLOv8-face)
            faces = self.detect_faces(frame)

            if len(faces) > 0:
                frames_with_faces += 1

            # 2. Classificar emoções (DeepFace)
            for face in faces:
                try:
                    emotion_result = self.classify_emotion(face['face_img'])

                    face['emotion'] = emotion_result['emotion']
                    face['emotion_conf'] = emotion_result['confidence']
                    face['probabilities'] = emotion_result['probabilities']

                    emotion_counts[emotion_result['emotion']] += 1
                    total_detections += 1

                except Exception as e:
                    # Em caso de erro, usar neutro
                    face['emotion'] = 'neutro'
                    face['emotion_conf'] = 0.5
                    face['probabilities'] = {}

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
            'emotion_distribution': emotion_counts,
            'dominant_emotion': dominant_emotion,
            'avg_faces_per_frame': total_detections / total_frames if total_frames > 0 else 0
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
            face_conf = det.get('confidence', 0.0)

            # Cor e emoji por emoção
            emoji, color = self.EMOTIONS.get(emotion, self.EMOTIONS['neutro'])

            # ═══════════════════════════════════════════════
            # DESENHAR ANOTAÇÕES
            # ═══════════════════════════════════════════════

            # 1. Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 2. Label
            if show_scores:
                label = f"{emoji} {emotion.capitalize()} ({conf*100:.0f}%)"
            else:
                label = f"{emoji} {emotion.capitalize()}"

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

            # 4. Mini-barra de confiança (opcional)
            if show_scores and 'probabilities' in det:
                self._draw_emotion_bar(annotated, det, x1, y2)

        # 5. Contador de pessoas
        people_count = len(detections)
        cv2.putText(
            annotated,
            f"👥 Pessoas: {people_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            cv2.LINE_AA
        )

        # Fundo do contador
        cv2.rectangle(annotated, (5, 5), (200, 40), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            f"👥 Pessoas: {people_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            cv2.LINE_AA
        )

        return annotated

    def _draw_emotion_bar(
        self,
        frame: np.ndarray,
        detection: Dict,
        x: int,
        y: int
    ):
        """
        Desenha mini-barra com top 3 emoções.

        Args:
            frame: Frame a anotar
            detection: Detecção com probabilidades
            x, y: Posição inicial
        """
        probs = detection.get('probabilities', {})
        if not probs:
            return

        # Top 3 emoções
        sorted_emotions = sorted(
            probs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        bar_width = 150
        bar_height = 15
        y_offset = y + 10

        for emotion, prob in sorted_emotions:
            emoji, color = self.EMOTIONS.get(emotion, ('?', (128, 128, 128)))

            # Barra de fundo
            cv2.rectangle(
                frame,
                (x, y_offset),
                (x + bar_width, y_offset + bar_height),
                (50, 50, 50), -1
            )

            # Barra de progresso
            filled_width = int(bar_width * prob)
            cv2.rectangle(
                frame,
                (x, y_offset),
                (x + filled_width, y_offset + bar_height),
                color, -1
            )

            # Texto
            text = f"{emoji} {prob*100:.0f}%"
            cv2.putText(
                frame, text,
                (x + 5, y_offset + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

            y_offset += bar_height + 5
