import torch
from transformers import AutoModel, AutoProcessor
from typing import List, Tuple
from torchvision import transforms
from collections import defaultdict
from controldetecting.CapProcessor import CapProcessor
from controldetecting.FrameAnnotator import FrameAnnotator

import time

class BehaviorClassifier:
    """
    Атрибуты:
        cap_processor (CapProcessor): Обработчик для подготовки кадров.
        frame_annotator (FrameAnnotator): Аннотатор для добавления меток на кадры.
        device (str): Устройство для выполнения вычислений (например, 'cuda' или 'cpu').
        model (AutoModel): Предварительно обученная модель для классификации.
        processor (AutoProcessor): Процессор для подготовки входных данных.
        transform (transforms.Compose): Трансформации для подготовки изображений.
        labels (List[str]): Список меток для классификации.
        label_to_colors (dict): Словарь для сопоставления меток с цветами.
        track_history (defaultdict): История треков для каждого объекта.
        track_ids_to_infer (List): Список идентификаторов треков для классификации.
        crops_to_infer (List): Список кропов для классификации.
        pred_labels (List[List[str]]): Предсказанные метки.
        pred_confs (List[List[float]]): Доверительные значения для предсказанных меток.
        num_video_sequence_samples (int): Количество образцов для классификации видео.
        video_cls_overlap_ratio (float): Коэффициент перекрытия для классификации видео.
        skip_frame (int): Количество кадров для пропуска.
        processed_box (List): Список обработанных боксов.
    """
    def __init__(self, cap_processor: CapProcessor, frame_annotator: FrameAnnotator, device, model_name, processor_name):
        self.device = device
        self.cap_processor = cap_processor
        self.frame_annotator = frame_annotator

        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.labels = [
            "calm pedestrian", "social interaction", "street vendor", "public transportation user", "recreational activity",
            "physical altercation", "aggressive gestures", "property damage", "harassment",
            "loitering", "street performer", "protesting"
                       ]

        self.label_to_colors = {
            "calm pedestrian": (0, 255, 0),  # Зеленый
            "social interaction": (0, 255, 0),
            "street vendor": (0, 255, 0),
            "public transportation user": (0, 255, 0),
            "recreational activity": (0, 255, 0),
            "physical altercation": (0, 0, 255), # Красный
            "aggressive gestures": (0, 0, 255),
            "property damage": (0, 0, 255),
            "harassment": (0, 0, 255),
            "loitering": (0, 255, 255),  # Желтый
            "street performer": (0, 255, 255),
            "protesting": (0, 255, 255),
        }

        # Инициализация истории треков и других вспомогательных структур
        self.track_history = defaultdict(list)
        self.track_ids_to_infer, self.crops_to_infer = [], []
        self.pred_labels, self.pred_confs = [], []
        self.num_video_sequence_samples = 8
        self.video_cls_overlap_ratio: float = 0.25
        self.skip_frame = 3

        self.processed_box = []

    def try_reset_to_infer(self, frame_counter):
        """
        Сбрасывает списки кропов и идентификаторов треков для классификации, если текущий кадр соответствует условию пропуска.

        Параметры:
            frame_counter (int): Счетчик кадров.
        """
        if frame_counter % self.skip_frame == 0:
            self.crops_to_infer = []
            self.track_ids_to_infer = []

    def __call__(self, frame, box, track_id, frame_counter):
        """
        Обрабатывает кадр для классификации поведения.

        Параметры:
            frame (np.ndarray): Текущий кадр.
            box (Tuple[int, int, int, int]): Координаты бокса.
            track_id (int): Идентификатор трека.
            frame_counter (int): Счетчик кадров.
        """
        self.processed_box.append(box)
        track_by_id = self.track_history[track_id]
        frame_mod_skip = frame_counter % self.skip_frame

        can_add = self.can_add_track_and_crop_to_infer(box, frame, frame_mod_skip, track_by_id)
        if can_add:
            corps = self.process_frame(track_by_id)
            self.crops_to_infer.append(corps)
            self.track_ids_to_infer.append(track_id)

        if self.can_classify_behavior(self.crops_to_infer, frame_counter, self.pred_labels, self.skip_frame):
            crops_batch = torch.cat(self.crops_to_infer, dim=0)
            start_inference_time = time.time()
            output_batch = self.predict(crops_batch)
            end_inference_time = time.time()
            inference_time = end_inference_time - start_inference_time
            print(f"video cls inference time: {inference_time:.4f} seconds")
            self.pred_labels, self.pred_confs = self.postprocess(output_batch)

    def annotate_frame(self, boxes, frame, classes):
        """
        Аннотирует кадр с предсказанными метками.

        Параметры:
            boxes (List[Tuple[int, int, int, int]]): Список боксов.
            frame (np.ndarray): Текущий кадр.
            classes (List[int]): Список классов.
        """
        if self.track_ids_to_infer and self.crops_to_infer:
            zipped_data = zip(boxes, self.pred_labels, self.pred_confs, classes)
            self.frame_annotator(frame, zipped_data, self.label_to_colors)
            self.processed_box.clear()

    def predict(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Выполняет предсказание на основе входных данных.

        Параметры:
            sequences (torch.Tensor): Входные данные для модели.

        Возвращает:
            torch.Tensor: Логиты для каждого видео.
        """
        input_ids = self.processor(text=self.labels, return_tensors="pt", padding=True)["input_ids"].to(self.device)
        inputs = {"pixel_values": sequences, "input_ids": input_ids}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs.logits_per_video

    def postprocess(self, outputs: torch.Tensor) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Обрабатывает выходные данные модели для получения меток и доверительных значений.

        Параметры:
            outputs (torch.Tensor): Логиты модели.

        Возвращает:
            Tuple[List[List[str]], List[List[float]]]: Предсказанные метки и доверительные значения.
        """
        pred_labels = []
        pred_confs = []
        with torch.no_grad():
            logits_per_video = outputs
            probs = logits_per_video.softmax(dim=-1)
        for prob in probs:
            top2_indices = prob.topk(2).indices.tolist()
            top2_labels = [self.labels[idx] for idx in top2_indices]
            top2_confs = prob[top2_indices].tolist()
            pred_labels.append(top2_labels)
            pred_confs.append(top2_confs)
        return pred_labels, pred_confs

    def can_add_track_and_crop_to_infer(self, box, frame, frame_mod_skip, track_by_id):
        """
        Проверяет, можно ли добавить трек и кроп для классификации.

        Параметры:
            box (Tuple[int, int, int, int]): Координаты бокса.
            frame (np.ndarray): Текущий кадр.
            frame_mod_skip (int): Остаток от деления счетчика кадров на количество пропускаемых кадров.
            track_by_id (List): История треков для текущего объекта.

        Возвращает:
            bool: True, если можно добавить трек и кроп, иначе False.
        """
        if frame_mod_skip == 0:
            crop = self.cap_processor.crop_and_pad(frame, box)
            track_by_id.append(crop)

        # если мы накопили больше кадров, чем надо (8), то выкинем, что бы не переполнить
        if len(track_by_id) > self.num_video_sequence_samples:
        # если накопили нужное количество кадров по выбранному объекту, то можно препроцессить кропс и ставить в очередь на классификацию
            track_by_id.pop(0)
        return len(track_by_id) == self.num_video_sequence_samples and frame_mod_skip == 0

    def can_classify_behavior(self, crops_to_infer, frame_counter, pred_labels, skip_frame):
        """
        Проверяет, можно ли классифицировать поведение.

        Параметры:
            crops_to_infer (List): Список кропов для классификации.
            frame_counter (int): Счетчик кадров.
            pred_labels (List[List[str]]): Предсказанные метки.
            skip_frame (int): Количество кадров для пропуска.

        Возвращает:
            bool: True, если можно классифицировать поведение, иначе False.
        """
        return crops_to_infer and (
                not pred_labels
                or frame_counter % int(
            self.num_video_sequence_samples * skip_frame * (1 - self.video_cls_overlap_ratio)) == 0)

    def process_frame(self, track_by_id):
        """
        Обрабатывает кадр для классификации видео.

        Параметры:
            track_by_id (List): История треков для текущего объекта.

        Возвращает:
            torch.Tensor: Обработанные кропы.
        """
        start_time = time.time()
        crops = self.cap_processor.preprocess_crops_for_video_cls(track_by_id)
        end_time = time.time()
        preprocess_time = end_time - start_time
        print(f"video cls preprocess time: {preprocess_time:.4f} seconds")
        return crops