"""
Analisador Híbrido: Combina VideoMAE com Análise de Pose.

Usa o melhor de cada método:
- VideoMAE: Atividades dinâmicas (Dançando, Acenando, Caretas)
- Pose + Objetos: Atividades estáticas (Trabalhando, Lendo, Telefone)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

from .videomae_analyzer import VideoMAEAnalyzer
from .activity_analyzer import ActivityAnalyzer


class HybridAnalyzer:
    """
    Analisador híbrido que combina VideoMAE e análise de pose.

    Estratégia:
    1. Usa VideoMAE para detectar atividades dinâmicas
    2. Usa análise de pose para atividades estáticas com objetos
    3. Prioriza o método mais confiável para cada tipo de atividade
    """

    # Atividades que VideoMAE detecta melhor
    VIDEOMAE_ACTIVITIES = {'Dançando', 'Acenando', 'Fazendo Caretas', 'Dando Gargalhadas'}

    # Atividades que análise de pose detecta melhor
    POSE_ACTIVITIES = {'Trabalhando', 'Lendo', 'Telefone'}

    def __init__(
        self,
        device: str = "auto",
        videomae_confidence: float = 0.3,
        pose_confidence: float = 0.5,
        pose_model: str = "models/yolo11x-pose.pt",
        object_model: str = "models/yolo11x.pt"
    ):
        """
        Inicializa o analisador híbrido.

        Args:
            device: 'cuda', 'cpu', ou 'auto'
            videomae_confidence: Threshold para VideoMAE
            pose_confidence: Threshold para análise de pose
            pose_model: Modelo YOLO para detecção de pose
            object_model: Modelo YOLO para detecção de objetos
        """
        print(f"🔀 Inicializando Analisador Híbrido...")

        # Inicializar VideoMAE
        self.videomae = VideoMAEAnalyzer(
            device=device,
            confidence_threshold=videomae_confidence
        )

        # Inicializar analisador de pose
        self.pose_analyzer = ActivityAnalyzer(
            pose_model_path=pose_model,
            object_model_path=object_model,
            confidence_threshold=pose_confidence,
            device=device
        )

        print(f"✅ Analisador Híbrido pronto!")
        print(f"   • VideoMAE: {', '.join(self.VIDEOMAE_ACTIVITIES)}")
        print(f"   • Pose+Objetos: {', '.join(self.POSE_ACTIVITIES)}")

    def process_scene(
        self,
        scene_path: str,
        output_path: str,
        clip_duration: float = 2.0,
        overlap: float = 1.0
    ) -> Dict[str, Any]:
        """
        Processa uma cena usando abordagem híbrida.

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
        method_used = []  # Rastrear qual método foi usado

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

            # Salvar clipe temporário para VideoMAE
            temp_clip = f"temp_clip_{start_frame}.mp4"
            temp_out = cv2.VideoWriter(temp_clip, fourcc, fps, (width, height))
            for f in frames_clip:
                temp_out.write(f)
            temp_out.release()

            # PASSO 1: Tentar VideoMAE
            videomae_result = self.videomae.process_video_clip(temp_clip)
            videomae_activity = videomae_result['activity']
            videomae_conf = videomae_result['confidence']

            # PASSO 2: Tentar análise de pose (frame do meio do clipe)
            middle_frame_idx = len(frames_clip) // 2
            middle_frame = frames_clip[middle_frame_idx]
            pose_frame_result = self.pose_analyzer.process_frame(middle_frame)

            # Extrair atividade com maior confiança de todas as pessoas
            pose_activity = 'Não Identificado'
            pose_conf = 0.0

            if pose_frame_result.get('people'):
                # Pegar a pessoa com maior confiança
                best_person = max(pose_frame_result['people'], key=lambda p: p.get('confidence', 0))
                pose_activity = best_person.get('activity', 'Não Identificado')
                pose_conf = best_person.get('confidence', 0.0)

            # Escolher melhor resultado baseado na estratégia híbrida
            final_activity, final_conf, method = self._select_best_result(
                videomae_activity, videomae_conf,
                pose_activity, pose_conf
            )

            activities.append(final_activity)
            method_used.append(method)

            # Anotar frames com resultado
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
            emoji = emoji_map.get(final_activity, '❓')

            # Indicador de método usado
            method_icon = '🤖' if method == 'VideoMAE' else '🎯'
            activity_text = f"{emoji} {final_activity} ({final_conf:.1%}) {method_icon}"

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

        # Calcular estatísticas de métodos usados
        method_counts = Counter(method_used)

        return {
            'scene_path': scene_path,
            'output_path': output_path,
            'dominant_activity': dominant,
            'activity_distribution': distribution,
            'method_distribution': dict(method_counts),
            'total_clips': len(activities),
            'fps': fps,
            'total_frames': total_frames
        }

    def _select_best_result(
        self,
        videomae_activity: str,
        videomae_conf: float,
        pose_activity: str,
        pose_conf: float
    ) -> tuple[str, float, str]:
        """
        Seleciona o melhor resultado entre VideoMAE e análise de pose.

        Estratégia:
        1. Se VideoMAE detectou atividade dinâmica com alta confiança -> usar VideoMAE
        2. Se pose detectou atividade estática -> usar pose
        3. Se ambos detectaram -> usar o método especializado para aquela atividade
        4. Caso contrário -> usar o resultado com maior confiança

        Args:
            videomae_activity: Atividade detectada pelo VideoMAE
            videomae_conf: Confiança do VideoMAE
            pose_activity: Atividade detectada pela análise de pose
            pose_conf: Confiança da análise de pose

        Returns:
            Tupla (atividade, confiança, método)
        """

        # Regra 1: VideoMAE detectou atividade dinâmica com boa confiança
        if videomae_activity in self.VIDEOMAE_ACTIVITIES and videomae_conf >= 0.3:
            return (videomae_activity, videomae_conf, 'VideoMAE')

        # Regra 2: Pose detectou atividade estática (especialidade dele)
        if pose_activity in self.POSE_ACTIVITIES and pose_conf >= 0.5:
            return (pose_activity, pose_conf, 'Pose')

        # Regra 3: Ambos detectaram a mesma categoria
        if videomae_activity == pose_activity and videomae_activity != 'Não Identificado':
            # Usar o método mais confiante
            if videomae_conf >= pose_conf:
                return (videomae_activity, videomae_conf, 'VideoMAE')
            else:
                return (pose_activity, pose_conf, 'Pose')

        # Regra 4: Usar o método especializado para cada atividade
        if videomae_activity in self.VIDEOMAE_ACTIVITIES:
            return (videomae_activity, videomae_conf, 'VideoMAE')

        if pose_activity in self.POSE_ACTIVITIES:
            return (pose_activity, pose_conf, 'Pose')

        # Regra 5: Usar o resultado com maior confiança
        if videomae_conf >= pose_conf and videomae_activity != 'Não Identificado':
            return (videomae_activity, videomae_conf, 'VideoMAE')
        elif pose_activity != 'Não Identificado':
            return (pose_activity, pose_conf, 'Pose')
        else:
            return ('Não Identificado', 0.0, 'Nenhum')
