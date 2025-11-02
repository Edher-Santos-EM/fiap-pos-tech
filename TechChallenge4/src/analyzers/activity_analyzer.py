"""
Analisador de atividades humanas.

ETAPA 3: Interpretação de Atividades
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
from ..activities.reading_activity import ReadingActivity
from ..activities.phone_activity import PhoneActivity
from ..activities.working_activity import WorkingActivity
from ..activities.dancing_activity import DancingActivity
from ..activities.laughing_activity import LaughingActivity
from ..activities.unknown_activity import UnknownActivity
from ..utils.progress_bar import create_progress_bar
# EmotionAnalyzer não pode ser importado aqui (ambiente diferente - venv_emotions vs venv_activities)


class ActivityAnalyzer:
    """
    Orquestrador de análise de atividades.

    Detecta poses e objetos usando YOLO e classifica atividades
    usando detectores específicos.
    """

    def __init__(
        self,
        pose_model_path: str = None,
        object_model_path: str = None,
        confidence_threshold: float = 0.6,
        device: str = 'auto',
        sharpness_threshold: float = 50.0,
        use_emotions: bool = True  # NOVO: integrar emoções
    ):
        """
        Inicializa o analisador de atividades.

        Args:
            pose_model_path: Caminho para modelo YOLO pose
            object_model_path: Caminho para modelo YOLO de objetos
            confidence_threshold: Threshold de confiança
            device: Dispositivo ('auto', 'cuda', 'cpu')
            sharpness_threshold: Threshold de nitidez para filtrar imagens borradas/radiografias (padrão: 50.0)
            use_emotions: Se deve usar DeepFace para detectar emoções (padrão: True)
        """
        # Configurar dispositivo
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.confidence_threshold = confidence_threshold
        self.sharpness_threshold = sharpness_threshold
        self.pose_model_path = pose_model_path
        self.object_model_path = object_model_path
        # Desabilitar emoções - requer ambiente diferente (venv_emotions)
        self.use_emotions = False

        # Modelos serão carregados sob demanda
        self.yolo_pose = None
        self.yolo_detect = None
        self.emotion_analyzer = None  # Não disponível neste ambiente

        # Instanciar detectores de atividade
        self.activity_detectors = [
            LaughingActivity(confidence_threshold),  # Gargalhada
            WorkingActivity(confidence_threshold),  # Laptop tem prioridade
            ReadingActivity(confidence_threshold),
            PhoneActivity(confidence_threshold),
            DancingActivity(confidence_threshold),  # Dança
            UnknownActivity(confidence_threshold)  # Sempre último (fallback)
        ]

        print(f"🚀 ActivityAnalyzer configurado para: {self.device.upper()}")
        print(f"   Detectores carregados: {len(self.activity_detectors)}")

    def _load_models(self):
        """Carrega modelos YOLO."""
        if self.yolo_pose is None:
            try:
                from ultralytics import YOLO

                # Modelo de pose
                if self.pose_model_path and Path(self.pose_model_path).exists():
                    print(f"📦 Carregando modelo de pose: {self.pose_model_path}")
                    self.yolo_pose = YOLO(self.pose_model_path)
                else:
                    # Se não existe, YOLO faz download automaticamente
                    print(f"📥 Baixando/carregando modelo: {self.pose_model_path}")
                    self.yolo_pose = YOLO(self.pose_model_path or 'yolov8x-pose.pt')

                self.yolo_pose.to(self.device)

                # Modelo de objetos
                if self.object_model_path and Path(self.object_model_path).exists():
                    print(f"📦 Carregando modelo de objetos: {self.object_model_path}")
                    self.yolo_detect = YOLO(self.object_model_path)
                else:
                    # Se não existe, YOLO faz download automaticamente
                    print(f"📥 Baixando/carregando modelo: {self.object_model_path}")
                    self.yolo_detect = YOLO(self.object_model_path or 'yolov8x.pt')

                self.yolo_detect.to(self.device)

                print("✅ Modelos YOLO carregados")

            except Exception as e:
                print(f"❌ Erro ao carregar modelos: {e}")
                self.yolo_pose = None
                self.yolo_detect = None

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Processa um frame detectando atividades de TODAS as pessoas.

        Args:
            frame: Frame do vídeo

        Returns:
            Dict com lista de pessoas e suas atividades detectadas
        """
        if self.yolo_pose is None or self.yolo_detect is None:
            self._load_models()

        if self.yolo_pose is None or self.yolo_detect is None:
            return {
                'people': [],
                'objects': [],
                'total_people': 0
            }

        try:
            # 1. Detectar poses de TODAS as pessoas
            pose_results = self.yolo_pose(frame, verbose=False)
            all_people_keypoints = []

            if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                kps = pose_results[0].keypoints.data
                # Processar TODAS as pessoas detectadas
                for person_kp in kps:
                    all_people_keypoints.append(person_kp.cpu().numpy())  # (17, 3)

            # 2. Detectar objetos
            obj_results = self.yolo_detect(frame, verbose=False)
            detected_objects = []

            for result in obj_results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    bbox = box.xyxy[0].cpu().numpy()

                    detected_objects.append({
                        'class': result.names[cls],
                        'confidence': conf,
                        'bbox': bbox.tolist()
                    })

            # 3. Detectar emoções faciais - DESABILITADO
            # Emoções não podem ser detectadas aqui porque EmotionAnalyzer
            # requer MediaPipe/TensorFlow que estão em venv_emotions
            face_emotions = []
            # if self.use_emotions:
            #     if self.emotion_analyzer is None:
            #         try:
            #             self.emotion_analyzer = EmotionAnalyzer(device=self.device)
            #         except Exception as e:
            #             print(f"⚠️  Não foi possível carregar DeepFace: {e}")
            #             self.use_emotions = False
            #
            #     if self.emotion_analyzer:
            #         try:
            #             faces = self.emotion_analyzer.detect_faces(frame)
            #             for face in faces:
            #                 emotion_result = self.emotion_analyzer.classify_emotion(face['face_crop'])
            #                 face_emotions.append({
            #                     'bbox': face['bbox'],
            #                     'emotion': emotion_result.get('dominant_emotion', 'neutral'),
            #                     'emotion_confidence': emotion_result.get('confidence', 0.0),
            #                     'all_emotions': emotion_result.get('emotions', {})
            #                 })
            #         except Exception as e:
            #             # Silenciosamente ignorar erros de DeepFace
            #             pass

            # 4. Processar cada pessoa separadamente
            people_results = []

            for person_idx, pose_keypoints in enumerate(all_people_keypoints):
                # IMPORTANTE: Passar TODOS os objetos detectados para análise
                # Não filtrar por proximidade - objetos podem estar na mesa/perto da pessoa
                person_objects = detected_objects

                # Encontrar emoção facial correspondente a esta pessoa
                face_data = None
                if face_emotions:
                    # Pegar centro da cabeça da pessoa
                    nose = pose_keypoints[0]
                    # Verificar se nose é válido (confiança > 0.5)
                    if len(nose) >= 3 and nose[2] > 0.5:
                        # Encontrar face mais próxima
                        min_dist = float('inf')
                        for face_emotion in face_emotions:
                            face_bbox = face_emotion['bbox']
                            face_center_x = (face_bbox[0] + face_bbox[2]) / 2
                            face_center_y = (face_bbox[1] + face_bbox[3]) / 2
                            dist = np.sqrt((nose[0] - face_center_x)**2 + (nose[1] - face_center_y)**2)
                            if dist < min_dist and dist < 100:  # Menos de 100px de distância
                                min_dist = dist
                                face_data = face_emotion

                # Executar detectores de atividade para esta pessoa
                best_result = None
                best_confidence = 0.0

                for detector in self.activity_detectors:
                    try:
                        result = detector.detect(pose_keypoints, person_objects, face_data)

                        if result['confidence'] > best_confidence:
                            best_confidence = result['confidence']
                            best_result = result
                            best_result['activity'] = detector.activity_name
                            best_result['activity_icon'] = detector.activity_icon
                            best_result['activity_color'] = detector.color

                    except Exception as e:
                        continue

                if best_result is None:
                    # Usar UnknownActivity
                    unknown = self.activity_detectors[-1]
                    best_result = unknown.detect(pose_keypoints, person_objects)
                    best_result['activity'] = unknown.activity_name
                    best_result['activity_icon'] = unknown.activity_icon
                    best_result['activity_color'] = unknown.color

                # Adicionar dados desta pessoa
                best_result['person_id'] = person_idx
                best_result['keypoints'] = pose_keypoints  # Keypoints para tracking
                best_result['pose_keypoints'] = pose_keypoints  # Manter para retrocompatibilidade
                people_results.append(best_result)

            # Mesclar pessoas sobrepostas (quando uma está quase atrás da outra)
            people_results = self._merge_overlapping_people(people_results, iou_threshold=0.5)

            return {
                'people': people_results,
                'objects': detected_objects,
                'total_people': len(people_results)
            }

        except Exception as e:
            print(f"⚠️  Erro no processamento: {e}")
            return {
                'people': [],
                'objects': [],
                'total_people': 0
            }

    def _get_nearby_objects(self, pose_keypoints: np.ndarray, all_objects: List[Dict[str, Any]], distance_threshold: float = 150.0) -> List[Dict[str, Any]]:
        """
        Filtra objetos que estão próximos a uma pessoa específica.

        Args:
            pose_keypoints: Keypoints da pessoa
            all_objects: Todos os objetos detectados no frame
            distance_threshold: Distância máxima em pixels

        Returns:
            Lista de objetos próximos à pessoa
        """
        nearby_objects = []

        # Calcular centro da pessoa (média dos keypoints válidos)
        valid_keypoints = [kp[:2] for kp in pose_keypoints if kp[2] > 0.3]
        if not valid_keypoints:
            return all_objects  # Se não há keypoints válidos, retornar todos

        person_center = np.mean(valid_keypoints, axis=0)

        for obj in all_objects:
            # Calcular centro do objeto
            bbox = obj['bbox']
            obj_center = np.array([
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2
            ])

            # Calcular distância
            distance = np.linalg.norm(person_center - obj_center)

            if distance <= distance_threshold:
                nearby_objects.append(obj)

        return nearby_objects

    def _calculate_person_sharpness(self, frame: np.ndarray, pose_keypoints: np.ndarray) -> float:
        """
        Calcula a nitidez da região da pessoa para filtrar detecções borradas.

        Args:
            frame: Frame completo
            pose_keypoints: Keypoints da pessoa

        Returns:
            Score de nitidez (maior = mais nítido). Retorna 0 se não calculável.
        """
        try:
            # Obter bounding box da pessoa baseado nos keypoints
            valid_keypoints = [kp[:2] for kp in pose_keypoints if kp[2] > 0.3]
            if len(valid_keypoints) < 3:
                return 0.0

            keypoints_array = np.array(valid_keypoints)
            x_min = int(max(0, keypoints_array[:, 0].min() - 20))
            x_max = int(min(frame.shape[1], keypoints_array[:, 0].max() + 20))
            y_min = int(max(0, keypoints_array[:, 1].min() - 20))
            y_max = int(min(frame.shape[0], keypoints_array[:, 1].max() + 20))

            # Extrair região da pessoa
            person_roi = frame[y_min:y_max, x_min:x_max]

            if person_roi.size == 0 or person_roi.shape[0] < 10 or person_roi.shape[1] < 10:
                return 0.0

            # Converter para escala de cinza
            gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)

            # Calcular variância do Laplaciano (medida de nitidez)
            # Valores maiores = imagem mais nítida
            # Valores baixos = imagem borrada
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            return float(laplacian_var)

        except Exception as e:
            return 0.0

    def _check_person_quality(self, frame: np.ndarray, pose_keypoints: np.ndarray) -> Dict[str, Any]:
        """
        Verifica a qualidade da detecção de pessoa (nitidez, contraste).

        Args:
            frame: Frame completo
            pose_keypoints: Keypoints da pessoa

        Returns:
            Dict com 'is_valid', 'sharpness', 'reason'
        """
        # Calcular nitidez
        sharpness = self._calculate_person_sharpness(frame, pose_keypoints)

        # Usar threshold configurável
        # Valores típicos:
        # - 20-30: Filtrar apenas radiografias/imagens muito borradas
        # - 50: Filtrar imagens moderadamente borradas (padrão)
        # - 100+: Filtrar qualquer imagem não perfeitamente nítida

        if sharpness < self.sharpness_threshold:
            return {
                'is_valid': False,
                'sharpness': sharpness,
                'reason': f'Imagem borrada ou de baixa qualidade (nitidez: {sharpness:.1f})'
            }

        return {
            'is_valid': True,
            'sharpness': sharpness,
            'reason': 'Qualidade adequada'
        }

    def _calculate_bbox_from_keypoints(self, keypoints: np.ndarray) -> tuple:
        """
        Calcula bounding box a partir dos keypoints.

        Args:
            keypoints: Array (17, 3) com keypoints

        Returns:
            Tuple (x_min, y_min, x_max, y_max) ou None
        """
        valid_points = keypoints[keypoints[:, 2] > 0.3][:, :2]
        if len(valid_points) < 3:
            return None

        x_min = valid_points[:, 0].min()
        x_max = valid_points[:, 0].max()
        y_min = valid_points[:, 1].min()
        y_max = valid_points[:, 1].max()

        return (x_min, y_min, x_max, y_max)

    def _calculate_iou(self, box1: tuple, box2: tuple) -> float:
        """
        Calcula IoU (Intersection over Union) entre duas bounding boxes.

        Args:
            box1: (x_min, y_min, x_max, y_max)
            box2: (x_min, y_min, x_max, y_max)

        Returns:
            IoU score (0 a 1)
        """
        if box1 is None or box2 is None:
            return 0.0

        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Área de interseção
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
            return 0.0

        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)

        # Áreas das boxes
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        # União
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _merge_overlapping_people(self, people_list: list, iou_threshold: float = 0.5) -> list:
        """
        Mescla pessoas com alta sobreposição (IoU).

        Args:
            people_list: Lista de pessoas detectadas
            iou_threshold: Threshold de IoU para considerar mesma pessoa

        Returns:
            Lista de pessoas sem duplicatas
        """
        if len(people_list) <= 1:
            return people_list

        # Calcular bounding boxes para todas as pessoas
        bboxes = []
        for person in people_list:
            keypoints = person.get('keypoints')
            if keypoints is not None:
                bbox = self._calculate_bbox_from_keypoints(keypoints)
                bboxes.append(bbox)
            else:
                bboxes.append(None)

        # Encontrar pares com alta sobreposição
        to_remove = set()
        for i in range(len(people_list)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(people_list)):
                if j in to_remove:
                    continue

                iou = self._calculate_iou(bboxes[i], bboxes[j])

                # Se IoU alto, são a mesma pessoa
                if iou > iou_threshold:
                    # Manter a pessoa com maior confiança
                    conf_i = people_list[i].get('confidence', 0)
                    conf_j = people_list[j].get('confidence', 0)

                    if conf_i >= conf_j:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break

        # Retornar apenas pessoas não marcadas para remoção
        return [p for idx, p in enumerate(people_list) if idx not in to_remove]

    def _get_head_position(self, pose_keypoints: np.ndarray) -> np.ndarray:
        """
        Extrai a posição da cabeça (nariz - keypoint 0) de uma pessoa.

        Args:
            pose_keypoints: Array com 17 keypoints [x, y, conf]

        Returns:
            Array [x, y] com a posição da cabeça, ou None se não detectado
        """
        # Keypoint 0 = nariz (centro da cabeça)
        nose = pose_keypoints[0]
        if nose[2] > 0.3:  # Confiança mínima
            return nose[:2]
        return None

    def _match_person_to_existing(self, head_pos: np.ndarray, people_tracking: dict, frame_number: int, max_distance: float = 100.0) -> int:
        """
        Associa uma posição de cabeça detectada a uma pessoa já rastreada,
        ou cria um novo ID se for uma pessoa nova.

        Args:
            head_pos: Posição [x, y] da cabeça
            people_tracking: Dicionário de tracking
            frame_number: Número do frame atual
            max_distance: Distância máxima em pixels para considerar a mesma pessoa

        Returns:
            ID da pessoa (existente ou novo)
        """
        if head_pos is None:
            return -1  # Cabeça não detectada

        best_match_id = -1
        best_distance = float('inf')

        # Procurar pessoa mais próxima nos últimos frames
        for person_id, data in people_tracking.items():
            if len(data['head_positions']) == 0:
                continue

            # Considerar as últimas N posições (não só a última) para melhor tracking
            # Isso ajuda com pessoas em movimento rápido
            recent_positions = data['head_positions'][-3:]  # Últimas 3 posições
            recent_frames = data['frames'][-3:]

            # Calcular distância mínima às posições recentes
            min_distance = float('inf')
            for i, pos in enumerate(recent_positions):
                dist = np.linalg.norm(head_pos - pos)
                if dist < min_distance:
                    min_distance = dist

            # Gap temporal desde última detecção
            last_frame = data['frames'][-1]
            frame_gap = frame_number - last_frame

            # Penalizar menos por gap (pessoas dançando podem "desaparecer" temporariamente)
            distance_penalty = min_distance + (frame_gap * 2)  # Reduzido de 5 para 2

            if distance_penalty < best_distance and min_distance < max_distance:
                best_distance = distance_penalty
                best_match_id = person_id

        return best_match_id

    def process_scene(
        self,
        scene_path: str,
        output_path: str,
        show_skeleton: bool = True,
        analysis_fps: int = 5  # Analisar 5 frames por segundo (não todos)
    ) -> Dict[str, Any]:
        """
        Processa cena completa, detectando e anotando atividades.

        Args:
            scene_path: Caminho da cena
            output_path: Caminho de saída
            show_skeleton: Mostrar skeleton de pose

        Returns:
            Dict com estatísticas da análise
        """
        cap = cv2.VideoCapture(scene_path)

        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir: {scene_path}")

        # Propriedades
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Criar writer
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Calcular intervalo de frames para análise
        frame_interval = max(1, int(fps / analysis_fps))

        # Estatísticas gerais
        activity_counts = {}
        total_frames_processed = 0
        total_frames_analyzed = 0

        # Abordagem simplificada: rastrear por índice de pessoa no frame
        people_tracking = {}  # {person_index: {activities: [], total_detections: 0}}
        all_detected_objects = set()  # Objetos detectados ao longo da cena
        max_people_in_frame = 0  # Número máximo de pessoas em qualquer frame

        pbar = create_progress_bar(
            total=total_frames,
            desc=f"🏃 Analisando atividades (1 a cada {frame_interval} frames)",
            unit="frame",
            colour="magenta"
        )

        frame_number = 0
        last_analysis_result = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Decidir se analisa este frame
            should_analyze = (frame_number % frame_interval == 0)

            if should_analyze:
                # Processar frame (agora retorna múltiplas pessoas)
                result = self.process_frame(frame)
                last_analysis_result = result
                total_frames_analyzed += 1

                # Agregar objetos detectados
                for obj in result.get('objects', []):
                    all_detected_objects.add(obj['class'])

                # Abordagem simplificada: usar índice da pessoa no frame
                people = result.get('people', [])
                num_people = len(people)

                # Atualizar máximo de pessoas
                if num_people > max_people_in_frame:
                    max_people_in_frame = num_people

                # Rastrear atividades por índice de pessoa
                for person_idx, person in enumerate(people):
                    if person_idx not in people_tracking:
                        people_tracking[person_idx] = {
                            'activities': [],
                            'total_detections': 0
                        }

                    activity = person.get('activity', 'Não Identificado')
                    people_tracking[person_idx]['activities'].append(activity)
                    people_tracking[person_idx]['total_detections'] += 1

                    # Contar atividades globais
                    if activity not in activity_counts:
                        activity_counts[activity] = 0
                    activity_counts[activity] += 1
            else:
                # Usar último resultado de análise para anotar
                result = last_analysis_result if last_analysis_result else {'people': [], 'objects': []}

            # Anotar TODOS os frames (mesmo os não analisados)
            annotated_frame = self.annotate_frame(frame, result, show_skeleton)
            writer.write(annotated_frame)

            total_frames_processed += 1
            frame_number += 1
            pbar.update(1)

        cap.release()
        writer.release()
        pbar.close()

        # Heurística para consolidar pessoas em movimento:
        # Se max_people_in_frame <= 1, significa apenas 1 pessoa por vez
        # Neste caso, consolidar todos os índices em uma única pessoa
        if max_people_in_frame <= 1 and len(people_tracking) > 1:
            # Consolidar todas as detecções em uma única pessoa
            consolidated_activities = []
            consolidated_detections = 0

            for person_idx, data in people_tracking.items():
                consolidated_activities.extend(data['activities'])
                consolidated_detections += data['total_detections']

            # Contar atividades consolidadas
            activity_count = {}
            for act in consolidated_activities:
                activity_count[act] = activity_count.get(act, 0) + 1

            if activity_count:
                dominant = max(activity_count, key=activity_count.get)
                confidence = activity_count[dominant] / len(consolidated_activities)

                people_summary = [{
                    'person_id': 1,
                    'dominant_activity': dominant,
                    'confidence': confidence,
                    'activity_distribution': activity_count,
                    'frames_detected': len(consolidated_activities),
                    'total_detections': consolidated_detections
                }]
            else:
                people_summary = []

        else:
            # Usar número máximo de pessoas detectadas
            total_people = max(max_people_in_frame, len(people_tracking))

            # Gerar resumo de atividades por pessoa
            people_summary = []
            for person_idx in range(total_people):
                if person_idx in people_tracking:
                    data = people_tracking[person_idx]
                    activities = data['activities']

                    if activities:
                        # Contar atividades desta pessoa
                        activity_count = {}
                        for act in activities:
                            activity_count[act] = activity_count.get(act, 0) + 1

                        dominant = max(activity_count, key=activity_count.get)
                        confidence = activity_count[dominant] / len(activities)

                        people_summary.append({
                            'person_id': person_idx + 1,
                            'dominant_activity': dominant,
                            'confidence': confidence,
                            'activity_distribution': activity_count,
                            'frames_detected': len(activities),
                            'total_detections': data['total_detections']
                        })
                else:
                    # Pessoa não foi rastreada (não apareceu em frames analisados)
                    people_summary.append({
                        'person_id': person_idx + 1,
                        'dominant_activity': 'Não Identificado',
                        'confidence': 0.0,
                        'activity_distribution': {'Não Identificado': 0},
                        'frames_detected': 0,
                        'total_detections': 0
                    })

        # Atividade predominante geral
        dominant_activity = max(activity_counts, key=activity_counts.get) if activity_counts else 'Não Identificado'

        return {
            'scene_path': scene_path,
            'output_path': output_path,
            'total_frames': total_frames_processed,
            'frames_analyzed': total_frames_analyzed,
            'analysis_interval': frame_interval,
            'activity_distribution': activity_counts,
            'dominant_activity': dominant_activity,
            'people': people_summary,
            'total_people': len(people_summary),
            'all_objects_detected': list(all_detected_objects)
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        result: Dict[str, Any],
        show_skeleton: bool = True
    ) -> np.ndarray:
        """
        Anota frame com atividades de TODAS as pessoas detectadas.

        Args:
            frame: Frame original
            result: Resultado da detecção (com múltiplas pessoas)
            show_skeleton: Mostrar skeleton de pose

        Returns:
            Frame anotado
        """
        annotated = frame.copy()

        # Desenhar objetos detectados (usando OpenCV - só ASCII)
        for obj in result.get('objects', []):
            if obj['confidence'] > 0.5:
                x1, y1, x2, y2 = map(int, obj['bbox'])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(annotated, obj['class'], (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Processar cada pessoa detectada
        people = result.get('people', [])

        for person_idx, person in enumerate(people):
            pose_keypoints = person.get('pose_keypoints')

            # Desenhar skeleton se disponível
            if show_skeleton and pose_keypoints is not None:
                annotated = self._draw_skeleton(annotated, pose_keypoints)

        # Converter para PIL para desenhar texto UTF-8
        annotated = self._draw_utf8_labels(annotated, people)

        return annotated

    def _draw_utf8_labels(self, frame: np.ndarray, people: List[Dict[str, Any]]) -> np.ndarray:
        """
        Desenha labels com suporte UTF-8 (acentos e emojis) usando PIL.

        Args:
            frame: Frame em formato OpenCV (BGR)
            people: Lista de pessoas detectadas

        Returns:
            Frame com labels UTF-8
        """
        # Converter BGR (OpenCV) para RGB (PIL)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Tentar carregar fonte TrueType com suporte a emojis
        font = None
        font_small = None
        supports_emoji = False

        # Lista de fontes para tentar (em ordem de preferência)
        font_options = [
            ("seguiemj.ttf", 20, 24),  # Segoe UI Emoji (Windows 10+)
            ("seguisym.ttf", 20, 24),  # Segoe UI Symbol (Windows)
            ("NotoColorEmoji.ttf", 20, 24),  # Noto Color Emoji (Linux)
            ("AppleColorEmoji.ttf", 20, 24),  # Apple Color Emoji (Mac)
            ("segoeui.ttf", 20, 24),  # Segoe UI (sem emoji)
            ("arial.ttf", 20, 24),  # Arial (sem emoji)
        ]

        for font_name, size_regular, size_small in font_options:
            try:
                font = ImageFont.truetype(font_name, size_regular)
                font_small = ImageFont.truetype(font_name, size_small)
                # Fontes de emoji são as primeiras 4
                supports_emoji = font_options.index((font_name, size_regular, size_small)) < 4
                break
            except:
                continue

        # Fallback para fonte padrão
        if font is None:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
            supports_emoji = False

        # Desenhar label para cada pessoa
        for person_idx, person in enumerate(people):
            pose_keypoints = person.get('pose_keypoints')
            activity = person.get('activity', 'Não Identificado')
            confidence = person.get('confidence', 0.0)
            color_bgr = person.get('activity_color', (128, 128, 128))

            # Converter cor BGR para RGB
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

            # Encontrar posição do label
            label_pos = self._get_label_position(pose_keypoints, person_idx, len(people))
            x, y = label_pos

            # Criar label SEM emoji (emojis apenas no relatório)
            label = f"{activity} ({confidence*100:.0f}%)"

            # Calcular tamanho do texto
            bbox = draw.textbbox((x, y), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Desenhar fundo do texto
            padding = 5
            draw.rectangle(
                [(x - padding, y - text_height - padding),
                 (x + text_width + padding, y + padding)],
                fill=color_rgb
            )

            # Desenhar texto
            draw.text((x, y - text_height), label, font=font, fill=(255, 255, 255))

        # Contador de pessoas
        if len(people) > 0:
            counter_text = f"Pessoas: {len(people)}"
            draw.text((10, 10), counter_text, font=font_small, fill=(255, 255, 255))

        # Converter de volta para OpenCV (RGB para BGR)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _get_label_position(self, pose_keypoints: np.ndarray, person_idx: int, total_people: int) -> Tuple[int, int]:
        """
        Calcula posição do label para uma pessoa.
        Usa posição da cabeça (nariz) se disponível, senão usa posição padrão.
        """
        if pose_keypoints is not None and len(pose_keypoints) > 0:
            nose = pose_keypoints[0]
            if nose[2] > 0.3:  # Confiança válida
                # Colocar label acima da cabeça
                return (int(nose[0]) - 50, int(nose[1]) - 30)

        # Posição padrão (lado direito, escalonado)
        return (10, 60 + person_idx * 40)

    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Desenha skeleton de pose."""
        # Conexões COCO
        skeleton = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Braços
            (5, 11), (6, 12), (11, 12),  # Tronco
            (11, 13), (13, 15), (12, 14), (14, 16)  # Pernas
        ]

        # Desenhar keypoints
        for i, kp in enumerate(keypoints):
            x, y, conf = kp
            if conf > 0.3:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        # Desenhar conexões
        for start, end in skeleton:
            if start < len(keypoints) and end < len(keypoints):
                kp1 = keypoints[start]
                kp2 = keypoints[end]
                if kp1[2] > 0.3 and kp2[2] > 0.3:
                    pt1 = (int(kp1[0]), int(kp1[1]))
                    pt2 = (int(kp2[0]), int(kp2[1]))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        return frame
