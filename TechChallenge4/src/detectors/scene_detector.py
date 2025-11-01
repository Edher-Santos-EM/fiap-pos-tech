"""
Detector de cenas usando diferença de frames.

ETAPA 1: Separação de Cenas
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from scipy import signal
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


class HybridSceneDetector(SceneDetector):
    """
    Detector de cenas híbrido com múltiplas métricas.

    Combina três métodos de detecção para maior precisão:
    1. Diferença de pixels (40%)
    2. Comparação de histogramas (40%)
    3. Detecção de bordas (20%)

    Utiliza detecção adaptativa de picos baseada em análise estatística.
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str = "output/cenas",
        threshold: float = 25.0,
        min_scene_duration: float = 1.0,
        codec: str = 'mp4v',
        device: str = 'cpu',
        histogram_threshold: float = 0.3,
        edge_threshold: float = 0.2,
        sigma_multiplier: float = 2.5,
        pixel_weight: float = 0.4,
        histogram_weight: float = 0.4,
        edge_weight: float = 0.2
    ):
        """
        Inicializa o detector híbrido de cenas.

        Args:
            video_path: Caminho para o vídeo
            output_dir: Diretório de saída
            threshold: Threshold base para normalização de diferença de pixels
            min_scene_duration: Duração mínima de cena em segundos
            codec: Codec para salvar vídeos
            device: Dispositivo (não usado para detecção de cenas)
            histogram_threshold: Threshold para diferença de histograma (não usado com detecção adaptativa)
            edge_threshold: Threshold para diferença de bordas (não usado com detecção adaptativa)
            sigma_multiplier: Multiplicador de desvio padrão para detecção de picos (default: 2.5)
            pixel_weight: Peso da métrica de diferença de pixels (default: 0.4)
            histogram_weight: Peso da métrica de histograma (default: 0.4)
            edge_weight: Peso da métrica de bordas (default: 0.2)
        """
        super().__init__(
            video_path=video_path,
            output_dir=output_dir,
            threshold=threshold,
            min_scene_duration=min_scene_duration,
            codec=codec,
            device=device
        )

        self.histogram_threshold = histogram_threshold
        self.edge_threshold = edge_threshold
        self.sigma_multiplier = sigma_multiplier

        # Pesos das métricas (devem somar 1.0)
        total_weight = pixel_weight + histogram_weight + edge_weight
        self.pixel_weight = pixel_weight / total_weight
        self.histogram_weight = histogram_weight / total_weight
        self.edge_weight = edge_weight / total_weight

    def detect_scenes(self) -> List[Dict[str, Any]]:
        """
        Detecta cenas usando método híbrido.

        Sobrescreve o método da classe pai para usar detecção híbrida.

        Returns:
            Lista de dicionários com informações de cada cena
        """
        return self.detect_scenes_hybrid()

    def detect_scenes_hybrid(self) -> List[Dict[str, Any]]:
        """
        Combina múltiplos métodos para melhor precisão.

        Returns:
            Lista de dicionários com informações de cada cena
        """
        cap = cv2.VideoCapture(self.video_path)
        scenes = []
        scene_start = 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        prev_frame = None
        differences = []

        # Buffer mínimo: 1 segundo ou 30 frames (o que for maior)
        min_buffer = max(30, int(fps))

        pbar = create_progress_bar(
            total=total_frames,
            desc="🔍 Detectando cenas (híbrido)",
            unit="frame",
            colour="cyan"
        )

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # 1. Diferença de pixels (método original melhorado)
                pixel_diff = np.mean(cv2.absdiff(prev_frame, gray))
                # Normalizar usando threshold como referência
                pixel_diff_norm = min(pixel_diff / self.threshold, 1.0)

                # 2. Diferença de histograma
                hist_curr = cv2.calcHist([gray], [0], None, [64], [0, 256])
                hist_prev = cv2.calcHist([prev_frame], [0], None, [64], [0, 256])

                # Normalizar histogramas
                cv2.normalize(hist_curr, hist_curr)
                cv2.normalize(hist_prev, hist_prev)

                # Correlação (1 = idênticos, -1 = opostos)
                hist_corr = cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_CORREL)
                hist_diff = 1 - hist_corr  # Converter para diferença (0 = idênticos, 2 = opostos)
                hist_diff_norm = min(hist_diff / 2.0, 1.0)  # Normalizar para [0, 1]

                # 3. Detecção de bordas
                edges_curr = cv2.Canny(gray, 50, 150)
                edges_prev = cv2.Canny(prev_frame, 50, 150)

                # Calcular similaridade de bordas
                intersection = np.sum(edges_curr & edges_prev)
                union = np.sum(edges_curr | edges_prev)
                edge_similarity = intersection / max(union, 1)
                edge_diff = 1 - edge_similarity

                # Combinar métricas com pesos configuráveis
                combined_score = (
                    self.pixel_weight * pixel_diff_norm +
                    self.histogram_weight * hist_diff_norm +
                    self.edge_weight * edge_diff
                )

                differences.append(combined_score)

                # Detectar picos usando análise estatística adaptativa
                if len(differences) >= min_buffer:
                    # Usar janela deslizante para estatísticas
                    window = differences[-min_buffer:]
                    mean = np.mean(window)
                    std = np.std(window)

                    # Detectar pico significativo (threshold adaptativo)
                    if combined_score > mean + self.sigma_multiplier * std:
                        # Verificar duração mínima
                        if frame_idx - scene_start >= self.min_scene_frames:
                            scenes.append({
                                'start_frame': scene_start,
                                'end_frame': frame_idx - 1,
                                'confidence': float(combined_score),
                                'pixel_diff': float(pixel_diff_norm),
                                'histogram_diff': float(hist_diff_norm),
                                'edge_diff': float(edge_diff)
                            })
                            scene_start = frame_idx

            prev_frame = gray
            frame_idx += 1
            pbar.update(1)

        # Processar última cena
        if frame_idx - scene_start >= self.min_scene_frames:
            scenes.append({
                'start_frame': scene_start,
                'end_frame': frame_idx - 1,
                'confidence': 0.0  # Última cena não tem transição
            })

        cap.release()
        pbar.close()

        return self._add_scene_metadata(scenes, fps)

    def _add_scene_metadata(self, scenes: List[Dict[str, Any]], fps: float) -> List[Dict[str, Any]]:
        """
        Adiciona metadados às cenas detectadas.

        Args:
            scenes: Lista de cenas detectadas
            fps: Taxa de frames por segundo do vídeo

        Returns:
            Lista de cenas com metadados adicionados
        """
        for idx, scene in enumerate(scenes):
            scene['id'] = idx + 1
            scene['start_time'] = scene['start_frame'] / fps
            scene['end_time'] = scene['end_frame'] / fps
            scene['duration'] = scene['end_time'] - scene['start_time']
            scene['num_frames'] = scene['end_frame'] - scene['start_frame'] + 1
            scene['filename'] = f"cena_{idx+1:03d}.mp4"

        return scenes
