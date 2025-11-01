"""Utilitários de processamento de vídeo."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class VideoProcessor:
    """Utilitários para processamento de vídeo."""

    @staticmethod
    def get_video_properties(video_path: str) -> dict:
        """
        Obtém propriedades do vídeo.

        Args:
            video_path: Caminho para o vídeo

        Returns:
            Dict com propriedades do vídeo
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

        props = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }

        cap.release()
        return props

    @staticmethod
    def validate_video(video_path: str) -> bool:
        """Valida se arquivo é um vídeo válido."""
        if not Path(video_path).exists():
            return False

        cap = cv2.VideoCapture(video_path)
        is_valid = cap.isOpened()
        cap.release()

        return is_valid

    @staticmethod
    def save_frame(frame: np.ndarray, output_path: str):
        """Salva frame como imagem."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, frame)

    @staticmethod
    def create_video_writer(
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        codec: str = 'mp4v'
    ) -> cv2.VideoWriter:
        """
        Cria VideoWriter para salvar vídeo.

        Args:
            output_path: Caminho de saída
            fps: FPS do vídeo
            frame_size: (width, height)
            codec: Codec a usar

        Returns:
            VideoWriter configurado
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        return writer
