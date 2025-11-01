"""
Analisador de atividades usando VideoMAE (Video Masked Autoencoders).

Usa modelo pré-treinado do Hugging Face para classificar atividades em vídeo
de forma muito mais precisa que análise de pose.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from collections import Counter
from PIL import Image, ImageDraw, ImageFont


class VideoMAEAnalyzer:
    """
    Analisador de atividades usando VideoMAE.

    Vantagens sobre análise de pose:
    - Entende contexto temporal (múltiplos frames)
    - Reconhece 400+ atividades diferentes
    - Muito mais preciso
    - Não depende de detecção de pose/objetos
    """

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        device: str = "auto",
        confidence_threshold: float = 0.3
    ):
        """
        Inicializa o analisador VideoMAE.

        Args:
            model_name: Nome do modelo no Hugging Face
            device: 'cuda', 'cpu', ou 'auto'
            confidence_threshold: Confiança mínima para aceitar classificação
        """
        # Detectar dispositivo
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🤖 Carregando VideoMAE ({model_name})...")
        print(f"   Dispositivo: {self.device.upper()}")

        # Carregar processor e modelo
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.confidence_threshold = confidence_threshold

        # Mapeamento de atividades do Kinetics-400 para nossas categorias
        self.activity_mapping = {
            # Dança
            'dancing': 'Dançando',
            'dancing ballet': 'Dançando',
            'dancing charleston': 'Dançando',
            'zumba': 'Dançando',
            'salsa dancing': 'Dançando',
            'breakdancing': 'Dançando',

            # Trabalhando/Computador
            'using computer': 'Trabalhando',
            'typing': 'Trabalhando',
            'texting': 'Trabalhando',

            # Lendo
            'reading book': 'Lendo',
            'reading newspaper': 'Lendo',

            # Telefone
            'talking on cell phone': 'Telefone',
            'texting': 'Telefone',

            # Acenando
            'waving hand': 'Acenando',
            'shaking hands': 'Acenando',

            # Caretas/Expressões faciais
            'making faces': 'Fazendo Caretas',
            'sticking tongue out': 'Fazendo Caretas',
            'blowing nose': 'Fazendo Caretas',
            'making a face': 'Fazendo Caretas',

            # Gargalhada/Rindo
            'laughing': 'Dando Gargalhadas',
            'tickling': 'Dando Gargalhadas',
            'giggling': 'Dando Gargalhadas',
        }

        print(f"✅ VideoMAE carregado com sucesso!")

    def process_video_clip(
        self,
        video_path: str,
        num_frames: int = 16,
        sample_fps: int = 2
    ) -> Dict[str, Any]:
        """
        Processa um clipe de vídeo e classifica a atividade.

        Args:
            video_path: Caminho para o vídeo
            num_frames: Número de frames a usar (padrão: 16)
            sample_fps: FPS para amostragem de frames

        Returns:
            Dict com atividade detectada e confiança
        """
        # Carregar frames do vídeo
        frames = self._load_video_frames(video_path, num_frames, sample_fps)

        if frames is None or len(frames) < num_frames:
            return {
                'activity': 'Não Identificado',
                'confidence': 0.0,
                'raw_predictions': []
            }

        # Processar frames
        inputs = self.processor(list(frames), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inferência
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Obter predições
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top5_probs, top5_indices = torch.topk(probs, k=5, dim=-1)

        # Converter para nossas categorias
        predictions = []
        for prob, idx in zip(top5_probs[0], top5_indices[0]):
            label = self.model.config.id2label[idx.item()]
            mapped_activity = self._map_activity(label)
            predictions.append({
                'activity': mapped_activity,
                'raw_label': label,
                'confidence': prob.item()
            })

        # Selecionar melhor predição
        best = predictions[0]

        return {
            'activity': best['activity'],
            'confidence': best['confidence'],
            'raw_label': best['raw_label'],
            'all_predictions': predictions
        }

    def process_scene(
        self,
        scene_path: str,
        output_path: str,
        clip_duration: float = 2.0,
        overlap: float = 1.0
    ) -> Dict[str, Any]:
        """
        Processa uma cena completa em múltiplos clipes.

        Args:
            scene_path: Caminho da cena
            output_path: Caminho para salvar vídeo anotado
            clip_duration: Duração de cada clipe em segundos
            overlap: Overlap entre clipes em segundos

        Returns:
            Dict com resultados da análise
        """
        cap = cv2.VideoCapture(scene_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Preparar writer de vídeo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Calcular clipes
        clip_frames = int(clip_duration * fps)
        overlap_frames = int(overlap * fps)
        step = clip_frames - overlap_frames

        # Armazenar atividades detectadas
        activities = []

        # Processar em clipes
        for start_frame in range(0, total_frames, step):
            end_frame = min(start_frame + clip_frames, total_frames)

            # Extrair frames do clipe
            frames_clip = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_clip.append(frame)

            if len(frames_clip) < 8:  # Mínimo de frames
                continue

            # Salvar clipe temporário
            temp_clip = f"temp_clip_{start_frame}.mp4"
            temp_out = cv2.VideoWriter(temp_clip, fourcc, fps, (width, height))
            for f in frames_clip:
                temp_out.write(f)
            temp_out.release()

            # Analisar clipe
            result = self.process_video_clip(temp_clip)
            activities.append(result['activity'])

            # Anotar frames com resultado usando Pillow (suporta emojis)
            emoji_map = {
                'Trabalhando': '💻',
                'Lendo': '📖',
                'Telefone': '📱',
                'Dançando': '💃',
                'Acenando': '👋',
                'Fazendo Caretas': '😜',
                'Dando Gargalhadas': '😂',
                'Não Identificado': '❓'
            }
            emoji = emoji_map.get(result['activity'], '❓')
            activity_text = f"{emoji} {result['activity']} ({result['confidence']:.1%})"

            for frame in frames_clip:
                # Converter BGR para RGB para Pillow
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(pil_image)

                # Tentar carregar fonte que suporte emojis
                font = None
                emoji_fonts = [
                    "seguiemj.ttf",  # Windows Segoe UI Emoji
                    "NotoColorEmoji.ttf",  # Linux
                    "AppleColorEmoji.ttf",  # macOS
                    "C:\\Windows\\Fonts\\seguiemj.ttf",  # Windows caminho completo
                    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"  # Linux caminho completo
                ]

                for font_name in emoji_fonts:
                    try:
                        font = ImageFont.truetype(font_name, 40)
                        break
                    except:
                        continue

                # Se nenhuma fonte de emoji foi encontrada, usar Arial para texto
                if font is None:
                    try:
                        font = ImageFont.truetype("arial.ttf", 40)
                    except:
                        font = ImageFont.load_default()

                # Desenhar texto com fundo para melhor legibilidade
                text_bbox = draw.textbbox((0, 0), activity_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Fundo semi-transparente (retângulo preto)
                padding = 10
                draw.rectangle(
                    [(40, 40), (40 + text_width + padding*2, 40 + text_height + padding*2)],
                    fill=(0, 0, 0, 180)
                )

                # Texto em verde
                draw.text((50, 50), activity_text, font=font, fill=(0, 255, 0))

                # Converter de volta para BGR para OpenCV
                frame_annotated = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                out.write(frame_annotated)

            # Limpar arquivo temporário
            Path(temp_clip).unlink(missing_ok=True)

        cap.release()
        out.release()

        # Calcular atividade dominante
        if activities:
            activity_counts = Counter(activities)
            dominant = activity_counts.most_common(1)[0][0]
            distribution = {k: v for k, v in activity_counts.items()}
        else:
            dominant = 'Não Identificado'
            distribution = {}

        return {
            'scene_path': scene_path,
            'output_path': output_path,
            'dominant_activity': dominant,
            'activity_distribution': distribution,
            'total_clips': len(activities),
            'fps': fps,
            'total_frames': total_frames
        }

    def _load_video_frames(
        self,
        video_path: str,
        num_frames: int = 16,
        sample_fps: int = 2
    ) -> Optional[List[np.ndarray]]:
        """
        Carrega frames uniformemente espaçados de um vídeo.

        Args:
            video_path: Caminho do vídeo
            num_frames: Número de frames a extrair
            sample_fps: FPS para amostragem

        Returns:
            Lista de frames ou None se erro
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calcular índices dos frames a extrair
            frame_interval = max(1, int(fps / sample_fps))
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Converter BGR para RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

            cap.release()
            return frames if len(frames) == num_frames else None

        except Exception as e:
            print(f"❌ Erro ao carregar vídeo: {e}")
            return None

    def _map_activity(self, kinetics_label: str) -> str:
        """
        Mapeia label do Kinetics-400 para nossas categorias.

        Args:
            kinetics_label: Label original do modelo

        Returns:
            Categoria mapeada
        """
        label_lower = kinetics_label.lower()

        for key, value in self.activity_mapping.items():
            if key in label_lower:
                return value

        # Se não encontrou mapeamento, retorna "Não Identificado"
        return 'Não Identificado'
