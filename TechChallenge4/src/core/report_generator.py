"""Geração de relatórios em Markdown."""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class ReportGenerator:
    """Gera relatórios consolidados em Markdown."""

    @staticmethod
    def generate_scene_report(scenes_data: Dict[str, Any], output_path: str):
        """Gera relatório de separação de cenas."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 📊 Relatório de Separação de Cenas\n\n")
            f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Vídeo:** {scenes_data.get('video_source', 'N/A')}\n")
            f.write(f"**Total de Cenas:** {scenes_data.get('total_scenes', 0)}\n\n")
            f.write("---\n\n")

            f.write("## 🎬 Cenas Detectadas\n\n")
            for scene in scenes_data.get('scenes', []):
                f.write(f"### Cena {scene['id']:03d}\n")
                f.write(f"- **Duração:** {scene['duration']:.2f}s\n")
                f.write(f"- **Frames:** {scene['start_frame']} - {scene['end_frame']}\n")
                f.write(f"- **Arquivo:** `{scene['filename']}`\n\n")

    @staticmethod
    def generate_emotion_report(emotions_data: Dict[str, Any], output_path: str):
        """Gera relatório completo de análise de sentimentos."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        scenes = emotions_data.get('scenes', [])
        total_detections = emotions_data.get('total_detections', 0)

        # Calcular estatísticas globais
        all_emotions = {}
        total_frames = 0
        frames_with_faces = 0

        for scene in scenes:
            total_frames += scene.get('total_frames', 0)
            frames_with_faces += scene.get('frames_with_faces', 0)

            # Agregar contagens de emoções
            for emotion, count in scene.get('emotion_distribution', {}).items():
                all_emotions[emotion] = all_emotions.get(emotion, 0) + count

        # Emoção predominante global
        dominant_emotion = 'N/A'
        if all_emotions:
            dominant_emotion = max(all_emotions, key=all_emotions.get)

        # Taxa de detecção global
        detection_rate = (frames_with_faces / total_frames * 100) if total_frames > 0 else 0

        with open(output_path, 'w', encoding='utf-8') as f:
            # ═══════════════════════════════════════════════
            # CABEÇALHO
            # ═══════════════════════════════════════════════
            f.write("# 😊 Relatório de Análise de Sentimentos\n\n")
            f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # ═══════════════════════════════════════════════
            # RESUMO EXECUTIVO
            # ═══════════════════════════════════════════════
            f.write("## 📈 Resumo Executivo\n\n")
            f.write(f"- **Total de Cenas Analisadas:** {len(scenes)}\n")
            f.write(f"- **Total de Frames Processados:** {total_frames:,}\n")
            f.write(f"- **Total de Detecções de Faces:** {total_detections:,}\n")
            f.write(f"- **Taxa de Detecção de Faces:** {detection_rate:.1f}%\n")
            f.write(f"- **Emoção Predominante Global:** {dominant_emotion.capitalize() if dominant_emotion != 'N/A' else 'N/A'}\n\n")

            # ═══════════════════════════════════════════════
            # DISTRIBUIÇÃO GLOBAL DE EMOÇÕES
            # ═══════════════════════════════════════════════
            f.write("## 🎭 Distribuição Global de Emoções\n\n")

            if all_emotions:
                # Ordenar emoções por frequência
                sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)

                f.write("| Emoção | Contagem | Percentual | Barra |\n")
                f.write("|--------|----------|------------|-------|\n")

                for emotion, count in sorted_emotions:
                    percentage = (count / total_detections * 100) if total_detections > 0 else 0
                    bar_length = int(percentage / 2)  # Barra de até 50 caracteres
                    bar = "█" * bar_length

                    # Emoji por emoção
                    emoji_map = {
                        'feliz': '😊',
                        'triste': '😢',
                        'raiva': '😠',
                        'surpreso': '😨',
                        'neutro': '😐',
                        'medo': '😰',
                        'nojo': '🤢'
                    }
                    emoji = emoji_map.get(emotion, '❓')

                    f.write(f"| {emoji} {emotion.capitalize()} | {count:,} | {percentage:.1f}% | {bar} |\n")

                f.write("\n")
            else:
                f.write("*Nenhuma emoção detectada.*\n\n")

            # ═══════════════════════════════════════════════
            # DETALHES POR CENA
            # ═══════════════════════════════════════════════
            f.write("## 🎬 Análise Detalhada por Cena\n\n")

            if not scenes:
                f.write("*Nenhuma cena analisada.*\n\n")
            else:
                for idx, scene in enumerate(scenes, 1):
                    scene_name = Path(scene.get('scene_path', '')).stem
                    scene_dominant = scene.get('dominant_emotion', 'neutro')
                    scene_detections = scene.get('total_detections', 0)
                    scene_frames = scene.get('total_frames', 0)
                    scene_faces_frames = scene.get('frames_with_faces', 0)
                    scene_detection_rate = scene.get('detection_rate', 0)
                    avg_faces = scene.get('avg_faces_per_frame', 0)
                    max_people = scene.get('max_people', 0)
                    frame_data = scene.get('frame_data', [])

                    # Emoji da emoção predominante
                    emoji_map = {
                        'feliz': '😊',
                        'triste': '😢',
                        'raiva': '😠',
                        'surpreso': '😨',
                        'neutro': '😐',
                        'medo': '😰',
                        'nojo': '🤢'
                    }
                    dominant_emoji = emoji_map.get(scene_dominant, '❓')

                    f.write(f"### {idx}. {scene_name}\n\n")
                    f.write(f"**Emoção Predominante:** {dominant_emoji} {scene_dominant.capitalize()}\n\n")

                    f.write("**Estatísticas:**\n")
                    f.write(f"- Total de Frames: {scene_frames:,}\n")
                    f.write(f"- Frames com Faces: {scene_faces_frames:,}\n")
                    f.write(f"- Taxa de Detecção: {scene_detection_rate:.1f}%\n")
                    f.write(f"- Total de Detecções: {scene_detections:,}\n")
                    f.write(f"- Máximo de Pessoas Simultâneas: {max_people}\n")
                    f.write(f"- Média de Faces por Frame: {avg_faces:.2f}\n\n")

                    # Análise detalhada de pessoas
                    if frame_data and max_people > 0:
                        f.write("#### 👥 Detalhamento por Pessoa\n\n")

                        # Agregar dados por pessoa ao longo do tempo
                        # Para cada frame, identificamos person_id 1, 2, 3, etc.
                        person_emotions = {}  # {person_id: {emotion: count}}

                        for frame in frame_data:
                            for face in frame['faces']:
                                pid = face['person_id']
                                emotion = face['emotion']

                                if pid not in person_emotions:
                                    person_emotions[pid] = {}

                                person_emotions[pid][emotion] = person_emotions[pid].get(emotion, 0) + 1

                        # Gerar tabela de pessoas
                        f.write("| Pessoa | Emoção Predominante | Aparições | Confiança Média |\n")
                        f.write("|--------|---------------------|-----------|----------------|\n")

                        for pid in sorted(person_emotions.keys()):
                            emotions = person_emotions[pid]
                            total_appearances = sum(emotions.values())

                            # Emoção predominante desta pessoa
                            dominant_person_emotion = max(emotions, key=emotions.get)
                            dominant_emoji = emoji_map.get(dominant_person_emotion, '❓')

                            # Calcular confiança média
                            confidences = []
                            for frame in frame_data:
                                for face in frame['faces']:
                                    if face['person_id'] == pid:
                                        confidences.append(face['confidence'])

                            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                            f.write(f"| Pessoa {pid} | {dominant_emoji} {dominant_person_emotion.capitalize()} | {total_appearances} frames | {avg_confidence:.1f}% |\n")

                        f.write("\n")

                        # Detalhe de emoções por pessoa
                        f.write("**Distribuição de Emoções por Pessoa:**\n\n")

                        for pid in sorted(person_emotions.keys()):
                            f.write(f"**Pessoa {pid}:**\n\n")

                            emotions = person_emotions[pid]
                            total_person_detections = sum(emotions.values())

                            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

                            for emotion, count in sorted_emotions:
                                percentage = (count / total_person_detections * 100) if total_person_detections > 0 else 0
                                emoji = emoji_map.get(emotion, '❓')
                                bar_length = int(percentage / 5)  # Barra de até 20 caracteres
                                bar = "█" * bar_length
                                f.write(f"- {emoji} {emotion.capitalize()}: {count} ({percentage:.1f}%) {bar}\n")

                            f.write("\n")

                    else:
                        f.write("*Nenhuma pessoa detectada nesta cena.*\n\n")

                    # Distribuição de emoções na cena
                    scene_emotions = scene.get('emotion_distribution', {})
                    if scene_emotions:
                        f.write("**Distribuição de Emoções:**\n\n")
                        f.write("| Emoção | Contagem | % da Cena |\n")
                        f.write("|--------|----------|----------|\n")

                        sorted_scene_emotions = sorted(
                            scene_emotions.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )

                        for emotion, count in sorted_scene_emotions:
                            if count > 0:  # Só mostrar emoções detectadas
                                percentage = (count / scene_detections * 100) if scene_detections > 0 else 0
                                emoji = emoji_map.get(emotion, '❓')
                                f.write(f"| {emoji} {emotion.capitalize()} | {count:,} | {percentage:.1f}% |\n")

                        f.write("\n")

                    f.write("**Arquivo de Saída:**\n")
                    output_file = Path(scene.get('output_path', '')).name
                    f.write(f"- `{output_file}`\n\n")

                    f.write("---\n\n")

            # ═══════════════════════════════════════════════
            # INSIGHTS E OBSERVAÇÕES
            # ═══════════════════════════════════════════════
            f.write("## 💡 Insights e Observações\n\n")

            if total_detections == 0:
                f.write("⚠️ **Nenhuma face foi detectada no vídeo.**\n\n")
                f.write("Possíveis causas:\n")
                f.write("- Vídeo não contém pessoas visíveis\n")
                f.write("- Qualidade de imagem muito baixa\n")
                f.write("- Faces muito pequenas ou obstruídas\n")
                f.write("- Threshold de confiança muito alto\n\n")
            else:
                # Análise da taxa de detecção
                if detection_rate < 20:
                    f.write(f"⚠️ **Taxa de detecção baixa ({detection_rate:.1f}%):**\n")
                    f.write("- A maioria dos frames não contém faces detectáveis\n")
                    f.write("- Vídeo pode conter mais objetos/cenários do que pessoas\n\n")
                elif detection_rate > 80:
                    f.write(f"✅ **Alta taxa de detecção ({detection_rate:.1f}%):**\n")
                    f.write("- A maioria dos frames contém faces detectadas\n")
                    f.write("- Vídeo focado em pessoas/rostos\n\n")

                # Análise da emoção predominante
                if dominant_emotion != 'N/A':
                    dominant_count = all_emotions.get(dominant_emotion, 0)
                    dominant_pct = (dominant_count / total_detections * 100) if total_detections > 0 else 0

                    if dominant_pct > 60:
                        f.write(f"📊 **Emoção muito predominante:** {dominant_emotion.capitalize()} ({dominant_pct:.1f}%)\n")
                        f.write("- O vídeo tem um tom emocional consistente\n\n")
                    elif dominant_pct < 30:
                        f.write(f"📊 **Emoções balanceadas:** Nenhuma emoção domina fortemente\n")
                        f.write("- O vídeo apresenta variedade emocional\n\n")

                # Análise de variação entre cenas
                if len(scenes) > 1:
                    scene_dominants = [s.get('dominant_emotion', 'neutro') for s in scenes]
                    unique_emotions = len(set(scene_dominants))

                    if unique_emotions == 1:
                        f.write(f"🎭 **Consistência Emocional:** Todas as cenas têm a mesma emoção predominante\n\n")
                    else:
                        f.write(f"🎭 **Variação Emocional:** {unique_emotions} emoções diferentes predominam nas cenas\n")
                        f.write("- O vídeo apresenta transições emocionais\n\n")

            # ═══════════════════════════════════════════════
            # RODAPÉ
            # ═══════════════════════════════════════════════
            f.write("---\n\n")
            f.write("*Relatório gerado automaticamente pelo sistema de análise de sentimentos.*\n")
            f.write("*Tecnologia: MediaPipe (Face Detection) + DeepFace (Emotion Recognition)*\n")

    @staticmethod
    def generate_scene_activity_report(scene_data: Dict[str, Any], output_path: str):
        """Gera relatório detalhado de uma cena específica."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        scene_name = Path(scene_data.get('scene_path', '')).stem
        people = scene_data.get('people', [])
        total_frames = scene_data.get('total_frames', 0)
        frames_analyzed = scene_data.get('frames_analyzed', 0)
        analysis_interval = scene_data.get('analysis_interval', 1)
        all_objects = scene_data.get('all_objects_detected', [])

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# 🎬 Relatório da Cena: {scene_name}\n\n")
            f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Informações da análise
            f.write("## ⚙️ Informações da Análise\n\n")
            f.write(f"- **Total de Frames:** {total_frames:,}\n")
            f.write(f"- **Frames Analisados:** {frames_analyzed:,}\n")
            f.write(f"- **Intervalo de Análise:** 1 frame a cada {analysis_interval} frames\n")
            f.write(f"- **Taxa de Análise:** {(frames_analyzed/total_frames*100):.1f}% dos frames\n\n")

            # Objetos detectados na cena
            f.write("## 📦 Objetos Detectados na Cena (ao longo do tempo)\n\n")
            if all_objects:
                f.write("Os seguintes objetos foram detectados em algum momento da cena:\n\n")
                for obj in sorted(all_objects):
                    f.write(f"- `{obj}`\n")
                f.write("\n")
            else:
                f.write("*Nenhum objeto específico foi detectado.*\n\n")

            # Pessoas detectadas
            f.write(f"## 👥 Pessoas Detectadas: {len(people)}\n\n")

            if not people:
                f.write("*Nenhuma pessoa foi detectada nesta cena.*\n\n")
            else:
                # Tabela resumo
                f.write("### Resumo Geral\n\n")
                f.write("| Pessoa | Atividade Predominante | Confiança | Detecções |\n")
                f.write("|--------|------------------------|-----------|----------|\n")

                emoji_map = {
                    'Trabalhando': '💻',
                    'Lendo': '📖',
                    'Telefone': '📱',
                    'Dançando': '💃',
                    'Não Identificado': '❓'
                }

                for person in people:
                    person_id = person['person_id']
                    dominant = person['dominant_activity']
                    confidence = person['confidence']
                    detections = person['total_detections']
                    emoji = emoji_map.get(dominant, '❓')

                    f.write(f"| Pessoa {person_id} | {emoji} {dominant} | {confidence*100:.1f}% | {detections} |\n")

                f.write("\n")

                # Detalhamento por pessoa
                f.write("### Detalhamento por Pessoa\n\n")

                for person in people:
                    person_id = person['person_id']
                    dominant = person['dominant_activity']
                    confidence = person['confidence']
                    activity_dist = person['activity_distribution']
                    frames_detected = person['frames_detected']
                    detections = person['total_detections']
                    emoji = emoji_map.get(dominant, '❓')

                    f.write(f"#### 👤 Pessoa {person_id}\n\n")
                    f.write(f"**Atividade Predominante:** {emoji} **{dominant}** ({confidence*100:.1f}% de confiança)\n\n")

                    f.write("**Estatísticas:**\n")
                    f.write(f"- Frames onde foi detectada: {frames_detected}\n")
                    f.write(f"- Total de detecções: {detections}\n\n")

                    # Distribuição de atividades
                    f.write("**Distribuição de Atividades:**\n\n")
                    f.write("| Atividade | Ocorrências | Percentual |\n")
                    f.write("|-----------|-------------|------------|\n")

                    sorted_activities = sorted(activity_dist.items(), key=lambda x: x[1], reverse=True)
                    for activity, count in sorted_activities:
                        pct = (count / detections * 100) if detections > 0 else 0
                        emoji = emoji_map.get(activity, '❓')
                        f.write(f"| {emoji} {activity} | {count} | {pct:.1f}% |\n")

                    f.write("\n")

                    # Análise da atividade predominante
                    if dominant == 'Trabalhando':
                        f.write("**💡 Análise:** Esta pessoa foi identificada trabalhando. ")
                        if 'laptop' in all_objects:
                            f.write("Laptop foi detectado na cena.\n\n")
                        else:
                            f.write("A postura e posição das mãos indicam uso de laptop.\n\n")

                    elif dominant == 'Lendo':
                        f.write("**💡 Análise:** Esta pessoa foi identificada lendo. ")
                        if 'book' in all_objects or 'paper' in all_objects:
                            f.write("Objetos de leitura foram detectados na cena.\n\n")
                        else:
                            f.write("⚠️ Nenhum objeto de leitura foi detectado visualmente, mas a pose indica leitura.\n\n")

                    elif dominant == 'Telefone':
                        f.write("**💡 Análise:** Esta pessoa foi identificada usando telefone. ")
                        if 'cell phone' in all_objects or 'phone' in all_objects:
                            f.write("Celular foi detectado na cena.\n\n")
                        else:
                            f.write("A mão estava próxima à orelha (possível ligação).\n\n")

                    elif dominant == 'Dançando':
                        f.write("**💡 Análise:** Esta pessoa foi identificada dançando. ")
                        f.write("Detectada por movimento corporal amplo, ambos os braços elevados e postura dinâmica.\n\n")

                    elif dominant == 'Não Identificado':
                        f.write("**💡 Análise:** Não foi possível identificar a atividade específica desta pessoa. ")
                        f.write("Isso pode ocorrer quando a pessoa não está realizando nenhuma das atividades monitoradas.\n\n")

                    f.write("---\n\n")

            # Rodapé
            f.write("## 📝 Observações\n\n")
            f.write("- A análise foi feita por **timeframe** (não frame a frame) para maior eficiência\n")
            f.write("- Objetos são agregados ao longo de toda a cena (apareçam no início, meio ou fim)\n")
            f.write("- Cada pessoa é rastreada individualmente ao longo da cena\n")
            f.write("- A confiança indica a consistência da atividade ao longo do tempo\n\n")

            f.write("---\n\n")
            f.write("*Relatório gerado automaticamente pelo sistema de interpretação de atividades.*\n")

    @staticmethod
    def generate_activity_report(activities_data: Dict[str, Any], output_path: str):
        """Gera relatório de interpretação de atividades."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        scenes = activities_data.get('scenes', [])

        # Calcular estatísticas globais
        all_activities = {}
        total_frames = 0

        for scene in scenes:
            total_frames += scene.get('total_frames', 0)

            # Agregar contagens de atividades
            for activity, count in scene.get('activity_distribution', {}).items():
                all_activities[activity] = all_activities.get(activity, 0) + count

        # Atividade predominante global
        most_common = activities_data.get('most_common', 'N/A')
        total_detections = sum(all_activities.values())

        with open(output_path, 'w', encoding='utf-8') as f:
            # ═══════════════════════════════════════════════
            # CABEÇALHO
            # ═══════════════════════════════════════════════
            f.write("# 🎭 Relatório de Interpretação de Atividades\n\n")
            f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # ═══════════════════════════════════════════════
            # RESUMO EXECUTIVO
            # ═══════════════════════════════════════════════
            # Calcular total de pessoas únicas em todas as cenas
            total_people = sum(scene.get('total_people', 0) for scene in scenes)

            f.write("## 📈 Resumo Executivo\n\n")
            f.write(f"- **Total de Cenas Analisadas:** {len(scenes)}\n")
            f.write(f"- **Total de Pessoas Detectadas:** {total_people}\n")
            f.write(f"- **Total de Frames Processados:** {total_frames:,}\n")
            f.write(f"- **Total de Detecções de Atividades:** {total_detections:,}\n")
            f.write(f"- **Atividade Mais Comum:** {most_common}\n\n")

            # ═══════════════════════════════════════════════
            # DISTRIBUIÇÃO GLOBAL DE ATIVIDADES
            # ═══════════════════════════════════════════════
            f.write("## 📊 Distribuição Global de Atividades\n\n")

            if all_activities:
                # Ordenar atividades por frequência
                sorted_activities = sorted(all_activities.items(), key=lambda x: x[1], reverse=True)

                f.write("| Atividade | Contagem | Percentual | Barra |\n")
                f.write("|-----------|----------|------------|-------|\n")

                for activity, count in sorted_activities:
                    percentage = (count / total_detections * 100) if total_detections > 0 else 0
                    bar_length = int(percentage / 2)  # Barra de até 50 caracteres
                    bar = "█" * bar_length

                    # Emoji por atividade
                    emoji_map = {
                        'Trabalhando': '💻',
                        'Lendo': '📖',
                        'Telefone': '📱',
                        'Dançando': '💃',
                        'Não Identificado': '❓'
                    }
                    emoji = emoji_map.get(activity, '❓')

                    f.write(f"| {emoji} {activity} | {count:,} | {percentage:.1f}% | {bar} |\n")

                f.write("\n")
            else:
                f.write("*Nenhuma atividade detectada.*\n\n")

            # ═══════════════════════════════════════════════
            # DETALHES POR CENA
            # ═══════════════════════════════════════════════
            f.write("## 🎬 Análise Detalhada por Cena\n\n")

            if not scenes:
                f.write("*Nenhuma cena analisada.*\n\n")
            else:
                for idx, scene in enumerate(scenes, 1):
                    scene_name = Path(scene.get('scene_path', '')).stem
                    scene_dominant = scene.get('dominant_activity', 'Não Identificado')
                    scene_frames = scene.get('total_frames', 0)

                    # Emoji da atividade predominante
                    emoji_map = {
                        'Trabalhando': '💻',
                        'Lendo': '📖',
                        'Telefone': '📱',
                        'Dançando': '💃',
                        'Não Identificado': '❓'
                    }
                    dominant_emoji = emoji_map.get(scene_dominant, '❓')

                    scene_people = scene.get('total_people', 0)

                    f.write(f"### {idx}. {scene_name}\n\n")
                    f.write(f"**Atividade Predominante:** {dominant_emoji} {scene_dominant}\n\n")

                    f.write("**Estatísticas:**\n")
                    f.write(f"- Total de Pessoas: {scene_people}\n")
                    f.write(f"- Total de Frames: {scene_frames:,}\n\n")

                    # Distribuição de atividades na cena
                    scene_activities = scene.get('activity_distribution', {})
                    if scene_activities:
                        f.write("**Distribuição de Atividades:**\n\n")
                        f.write("| Atividade | Contagem | % da Cena |\n")
                        f.write("|-----------|----------|----------|\n")

                        sorted_scene_activities = sorted(
                            scene_activities.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )

                        scene_total = sum(scene_activities.values())
                        for activity, count in sorted_scene_activities:
                            if count > 0:  # Só mostrar atividades detectadas
                                percentage = (count / scene_total * 100) if scene_total > 0 else 0
                                emoji = emoji_map.get(activity, '🎯')
                                f.write(f"| {emoji} {activity} | {count:,} | {percentage:.1f}% |\n")

                        f.write("\n")

                    f.write("**Arquivo de Saída:**\n")
                    output_file = Path(scene.get('output_path', '')).name
                    f.write(f"- `{output_file}`\n\n")

                    f.write("---\n\n")

            # ═══════════════════════════════════════════════
            # INSIGHTS E OBSERVAÇÕES
            # ═══════════════════════════════════════════════
            f.write("## 💡 Insights e Observações\n\n")

            if total_detections == 0:
                f.write("⚠️ **Nenhuma atividade foi detectada no vídeo.**\n\n")
                f.write("Possíveis causas:\n")
                f.write("- Vídeo não contém pessoas visíveis\n")
                f.write("- Poses não são detectáveis pelo modelo\n")
                f.write("- Threshold de confiança muito alto\n\n")
            else:
                # Análise da atividade predominante
                if most_common != 'N/A' and all_activities:
                    dominant_count = all_activities.get(most_common, 0)
                    dominant_pct = (dominant_count / total_detections * 100) if total_detections > 0 else 0

                    if dominant_pct > 60:
                        f.write(f"📊 **Atividade muito predominante:** {most_common} ({dominant_pct:.1f}%)\n")
                        f.write("- O vídeo tem uma atividade consistente\n\n")
                    elif dominant_pct < 30:
                        f.write(f"📊 **Atividades balanceadas:** Nenhuma atividade domina fortemente\n")
                        f.write("- O vídeo apresenta variedade de atividades\n\n")

                # Análise de variação entre cenas
                if len(scenes) > 1:
                    scene_dominants = [s.get('dominant_activity', 'Não Identificado') for s in scenes]
                    unique_activities = len(set(scene_dominants))

                    if unique_activities == 1:
                        f.write(f"🎭 **Consistência:** Todas as cenas têm a mesma atividade predominante\n\n")
                    else:
                        f.write(f"🎭 **Variação:** {unique_activities} atividades diferentes predominam nas cenas\n")
                        f.write("- O vídeo apresenta transições entre atividades\n\n")

            # ═══════════════════════════════════════════════
            # RODAPÉ
            # ═══════════════════════════════════════════════
            f.write("---\n\n")
            f.write("*Relatório gerado automaticamente pelo sistema de interpretação de atividades.*\n")
            f.write("*Tecnologia: YOLOv8 (Pose + Object Detection)*\n")

    @staticmethod
    def generate_consolidated_report(all_data: Dict[str, Any], output_path: str):
        """Gera relatório consolidado final."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 🎬 Relatório Completo de Análise de Vídeo\n\n")
            f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            f.write("## 🎯 Sumário Executivo\n\n")
            f.write(f"- **Vídeo Analisado:** {all_data.get('video_path', 'N/A')}\n")
            f.write(f"- **Cenas Detectadas:** {all_data.get('total_scenes', 0)}\n")
            f.write(f"- **Tempo de Processamento:** {all_data.get('processing_time', 0):.1f}s\n\n")
