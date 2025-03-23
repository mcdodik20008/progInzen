import json
import os
import time
from collections import defaultdict
from datetime import datetime
import random
import cv2
import numpy as np


class ClipBufferEntry:
    def __init__(self):
        self.frames = []
        self.last_seen = time.time()
        self.clip_index = 0

    def add_frame(self, frame: np.ndarray):
        self.frames.append(frame)
        self.last_seen = time.time()

    def is_ready(self, target_size: int) -> bool:
        return len(self.frames) >= target_size

    def reset(self):
        self.frames.clear()

class TrackClipBuffer:
    """
    Буфер для накопления и сохранения клипов:
    - Сохраняет `.avi` и кадры `.jpg`
    - Обеспечивает уникальность клипов по track_id и clip_index
    - Очищает устаревшие буферы
    """

    def __init__(
        self,
        output_dir: str,
        clip_size: int = 16,
        frame_rate: int = 25,
        max_idle_time: float = 10.0  # сек — через сколько удалять старые треки
    ):
        self.output_dir = output_dir
        self.clip_size = clip_size
        self.frame_rate = frame_rate
        self.max_idle_time = max_idle_time

        self.buffers: dict[int, ClipBufferEntry] = {}
        self.global_clip_counter = 0
        self.strategies = {}  # track_id -> strategy
        self.frame_counters = {}  # track_id -> frame index
        self.frame_indices = defaultdict(list)  # track_id → список номеров кадров

        self.frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

    def add(self, track_id: int, frame: np.ndarray):
        if track_id not in self.buffers:
            self.buffers[track_id] = ClipBufferEntry()
            self.strategies[track_id] = self._get_strategy()
            self.frame_counters[track_id] = 0

        self.frame_counters[track_id] += 1
        frame_idx = self.frame_counters[track_id]
        strategy = self.strategies[track_id]

        if not self._should_add_frame(strategy, frame_idx):
            return

        self.frame_indices[track_id].append(frame_idx)

        buffer = self.buffers[track_id]
        buffer.add_frame(frame)

        if buffer.is_ready(self.clip_size):
            self._save_clip(track_id, buffer.frames, buffer.clip_index)
            buffer.clip_index += 1
            buffer.reset()
            self.global_clip_counter += 1

        self._cleanup_stale_tracks()

    @staticmethod
    def _should_add_frame(strategy: str, frame_idx: int) -> bool:
        if strategy == "uniform":
            return True
        elif strategy == "step":
            return frame_idx % 3 == 0
        elif strategy == "random_step":
            return random.choice([True, False, False, False, False])  # примерно каждый 5-й
        return True

    def _save_clip(self, track_id: int, frames: list[np.ndarray], clip_index: int):
        if not frames:
            return

        h, w, _ = frames[0].shape

        # Сохраняем кадры
        frame_dir = os.path.join(self.frames_dir, f"{self.global_clip_counter:05d}_track_{track_id}_part_{clip_index}")
        os.makedirs(frame_dir, exist_ok=True)
        for idx, frame in enumerate(frames):
            resized = cv2.resize(frame, (224, 224))
            frame_path = os.path.join(frame_dir, f"frame_{idx:02d}.jpg")
            cv2.imwrite(frame_path, resized)

        self._save_metadata(track_id, clip_index, frame_dir)
        print(f"[✓] Сохранён клип: {frame_dir} + кадры ({len(frames)} шт)")

    def _cleanup_stale_tracks(self):
        now = time.time()
        to_delete = [tid for tid, buf in self.buffers.items() if now - buf.last_seen > self.max_idle_time]
        for tid in to_delete:
            print(f"[i] Удаление устаревшего трека {tid}")
            del self.buffers[tid]

    def _save_metadata(self, track_id: int, clip_index: int, frame_dir: str):
        clip_id = f"{self.global_clip_counter:05d}_track_{track_id}_part_{clip_index}"
        metadata = {
            "clip_id": clip_id,
            "track_id": track_id,
            "clip_index": clip_index,
            "global_id": self.global_clip_counter,
            "timestamp": datetime.now().isoformat(),
            "strategy": self.strategies[track_id],
            "clip_size": self.clip_size,
            "frame_count": self.frame_counters[track_id],
            "frame_indices": self.frame_indices[track_id][-self.clip_size:],  # последние N
            "frame_size": [224, 224],
            "source": "yolo+track_id",
            "resized": True
        }

        metadata_path = os.path.join(frame_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _get_strategy(self) -> str:
        return random.choice(["uniform", "step", "random_step"])
