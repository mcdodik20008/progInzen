from ultralytics import YOLO
import numpy as np
from typing import List


class Detection:
    def __init__(self, bbox: List[float], confidence: float, class_id: int, class_name: str, track_id: int = -1):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.track_id = track_id

    def is_person(self) -> bool:
        return self.class_name == "person"


class FrameProcessor:
    """
    Выполняет детекцию объектов на кадре.
    """
    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.3):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Детектирует объекты на кадре.
        """
        results = self.model.track(frame, persist=True)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            track_id = int(box.id[0])
            if conf >= self.conf_threshold:
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                detections.append(
                    Detection(bbox=bbox, confidence=conf, class_id=class_id, class_name=class_name, track_id=track_id)
                )

        return detections

    def contains_person(self, detections: List[Detection]) -> bool:
        """
        Возвращает True, если на кадре есть хотя бы один человек.
        """
        return any(d.is_person() for d in detections)
