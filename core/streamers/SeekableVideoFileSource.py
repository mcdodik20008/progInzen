import cv2
import numpy as np
from typing import Generator

from core.video_stream import IVideoStreamSource


class SeekableVideoFileSource(IVideoStreamSource):
    """
    Чтение локального видео с заданного времени.
    """
    def __init__(self, video_path: str, start_time_sec: float = 0.0):
        self.video_path = video_path
        self.start_time_sec = start_time_sec
        self.cap = None
        self.frame_rate = None

    def start(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {self.video_path}")

        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(self.start_time_sec * self.frame_rate)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def frames(self) -> Generator[np.ndarray, None, None]:
        if self.cap is None:
            raise RuntimeError("Видео не запущено. Вызови start()")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def stop(self):
        if self.cap:
            self.cap.release()