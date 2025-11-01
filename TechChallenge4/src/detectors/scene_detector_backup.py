"""
Detector de cenas usando diferença de frames.

ETAPA 1: Separação de Cenas
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from ..core.video_processor import VideoProcessor
from ..utils.progress_bar import create_progress_bar


class SceneDetector:
    """
    Detecta mudanças de cena em vídeos.

    Utiliza análise de diferença entre frames consecutivos para identificar
    transições/cortes de cena.
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str = "output/cenas",
        threshold: float = 25.0,
        min_scene_duration: float = 1.0,
        codec: str = 'mp4v',
        device: str = 'cpu'
    ):
        """
        Inicializa o detector de cenas.

        Args:
            video_path: Caminho para o vídeo
            output_dir: Diretório de saída
            threshold: Threshold para detecção de mudança de cena
            min_scene_duration: Duração mínima de cena em segundos
            codec: Codec para salvar vídeos
            device: Dispositivo (não usado para detecção de cenas)
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.threshold = threshold
        self.min_scene_duration = min_scene_duration
        self.codec = codec

        # Validar vídeo
        if not VideoProcessor.validate_video(video_path):
            raise ValueError(f"Vídeo inválido: {video_path}")

        # Obter propriedades
        self.video_props = VideoProcessor.get_video_properties(video_path)
        self.min_scene_frames = int(self.video_props['fps'] * min_scene_duration)

        # Criar diretório de saída
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def detect_scenes(self) -> List[Dict[str, Any]]:
        """
        Detecta cenas no vídeo.

        Returns:
            Lista de dicionários com informações de cada cena
        """
        cap = cv2.VideoCapture(self.video_path)
        scenes = []
        scene_start = 0
        prev_frame = None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        pbar = create_progress_bar(
            total=total_frames,
            desc=f"🔍 Detectando cenas",
            unit="frame",
            colour="green"
        )

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Converter para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Comparar com frame anterior
            if prev_frame is not None:
                # Calcular diferença
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)

                # Detectar mudança de cena
                if mean_diff > self.threshold:
                    # Verificar duração mínima
                    if frame_idx - scene_start >= self.min_scene_frames:
                        scenes.append({
                            'start_frame': scene_start,
                            'end_frame': frame_idx - 1
                        })
                        scene_start = frame_idx

            prev_frame = gray
            frame_idx += 1
            pbar.update(1)

        # Adicionar última cena
        if frame_idx - scene_start >= self.min_scene_frames:
            scenes.append({
                'start_frame': scene_start,
                'end_frame': frame_idx - 1
            })

        cap.release()
        pbar.close()

        # Adicionar metadados
        for idx, scene in enumerate(scenes):
            scene['id'] = idx + 1
            scene['start_time'] = scene['start_frame'] / fps
            scene['end_time'] = scene['end_frame'] / fps
            scene['duration'] = scene['end_time'] - scene['start_time']
            scene['num_frames'] = scene['end_frame'] - scene['start_frame'] + 1
            scene['filename'] = f"cena_{idx+1:03d}.mp4"

        return scenes

    def save_scenes(self, scenes_info: List[Dict[str, Any]]) -> List[str]:
        """
        Salva cenas como vídeos individuais.

        Args:
            scenes_info: Informações das cenas

        Returns:
            Lista de caminhos dos vídeos salvos
        """
        saved_paths = []
        cap = cv2.VideoCapture(self.video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

        pbar = create_progress_bar(
            total=len(scenes_info),
            desc="💾 Salvando cenas",
            unit="cena",
            colour="blue"
        )

        for scene in scenes_info:
            output_path = self.output_dir / scene['filename']

            # Criar writer
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

            # Posicionar no início da cena
            cap.set(cv2.CAP_PROP_POS_FRAMES, scene['start_frame'])

            # Salvar frames da cena
            for _ in range(scene['num_frames']):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)

            writer.release()
            saved_paths.append(str(output_path))
            pbar.update(1)

        cap.release()
        pbar.close()

        return saved_paths

    def generate_thumbnails(self, scenes_info: List[Dict[str, Any]]) -> List[str]:
        """
        Gera thumbnails para cada cena (primeiro frame).

        Args:
            scenes_info: Informações das cenas

        Returns:
            Lista de caminhos dos thumbnails
        """
        thumbnail_dir = self.output_dir / "thumbnails"
        thumbnail_dir.mkdir(exist_ok=True)

        thumbnail_paths = []
        cap = cv2.VideoCapture(self.video_path)

        for scene in scenes_info:
            # Posicionar no primeiro frame da cena
            cap.set(cv2.CAP_PROP_POS_FRAMES, scene['start_frame'])
            ret, frame = cap.read()

            if ret:
                thumbnail_path = thumbnail_dir / f"cena_{scene['id']:03d}.jpg"
                cv2.imwrite(str(thumbnail_path), frame)
                thumbnail_paths.append(str(thumbnail_path))
                scene['thumbnail'] = str(thumbnail_path.relative_to(self.output_dir.parent))

        cap.release()
        return thumbnail_paths

    def export_metadata(self, scenes_info: List[Dict[str, Any]], output_path: str = None):
        """
        Exporta metadados das cenas para JSON.

        Args:
            scenes_info: Informações das cenas
            output_path: Caminho de saída (opcional)
        """
        if output_path is None:
            output_path = self.output_dir / "cenas_metadata.json"

        metadata = {
            'video_source': self.video_path,
            'total_scenes': len(scenes_info),
            'scenes': scenes_info,
            'video_properties': self.video_props
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Metadados salvos em: {output_path}")
