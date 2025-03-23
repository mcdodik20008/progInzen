import subprocess
from typing import Generator

import numpy as np

from core.video_stream import IVideoStreamSource


class VideoStreamSource(IVideoStreamSource):
    """
    Источник видеопотока с YouTube через yt-dlp и ffmpeg.
    Предоставляет кадры как генератор.
    """

    def __init__(self, youtube_url: str, width: int = 1280, height: int = 720):
        self.youtube_url = youtube_url
        self.width = width
        self.height = height
        self.process = None
        self.frame_size = self.width * self.height * 3  # RGB 3 байта на пиксель

    def start(self):
        """
        Запускает subprocess для захвата потока.
        """
        command = (
            f"yt-dlp -f best -o - {self.youtube_url} | "
            f"ffmpeg -i - -f rawvideo -pix_fmt bgr24 -"
        )
        self.process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Генератор кадров из видеопотока.
        """
        if self.process is None:
            raise RuntimeError("Stream not started. Call start() first.")

        while True:
            raw_frame = self.process.stdout.read(self.frame_size)
            if len(raw_frame) != self.frame_size:
                break
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
            yield frame

    def stop(self):
        """
        Завершает subprocess.
        """
        if self.process:
            self.process.terminate()
            self.process.wait()

