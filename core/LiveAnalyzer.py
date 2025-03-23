import os
import re
from typing import List

import cv2
from collections import defaultdict, deque

import numpy as np

from inference.behavior_classifier import BehaviorClassifier
from core.frame_processor import FrameProcessor


class LiveAnalyzer:
    def __init__(self, stream, clip_size=16):
        self.stream = stream
        self.processor = FrameProcessor()
        self.classifier = BehaviorClassifier(model_path=self.get_latest_model_path('./ready_models'))
        self.buffers = defaultdict(lambda: deque(maxlen=clip_size))
        self.predictions = {}  # track_id → (label, probs)

    def run(self):
        self.stream.start()

        for frame in self.stream.frames():
            detections = self.processor.detect(frame)

            for det in detections:
                if not det.is_person():
                    continue

                x1, y1, x2, y2 = map(int, det.bbox)
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue

                tid = det.track_id
                self.buffers[tid].append(cropped)

                # Классифицируем, если накопили достаточно кадров
                if len(self.buffers[tid]) == self.buffers[tid].maxlen:
                    clip = list(self.buffers[tid])
                    video = self._preprocess_clip(clip)
                    clip_frames = list(video)
                    label, probs = self.classifier(clip_frames)
                    self.predictions[tid] = (label, probs)

                # Отрисовка
                color = (0, 255, 0)
                label_text = self.predictions.get(tid, ("...", {}))[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Показываем результат
            cv2.imshow("LiveAnalyzer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.stream.stop()
        cv2.destroyAllWindows()


    def _preprocess_clip(self, frames: List[np.ndarray]) -> np.ndarray:
        resized = [cv2.resize(f, self.classifier.frame_size) for f in frames]
        video = np.array(resized, dtype=np.float32) / 255.0
        return video  # [T, H, W, C]

    def get_latest_model_path(self, base_dir: str = "ready_models") -> str:
        """
        Возвращает путь к последней (vN) модели в указанной директории.
        """
        versions = []
        for name in os.listdir(base_dir):
            match = re.match(r"v(\d+)$", name)
            if match:
                versions.append(int(match.group(1)))

        if not versions:
            raise FileNotFoundError("❌ Не найдены директории вида vN в 'ready_models'")

        latest = max(versions)
        model_path = os.path.join(base_dir, f"v{latest}", "model.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Модель не найдена по пути: {model_path}")

        print(f"[📦] Загружаем модель: v{latest}")
        return model_path
