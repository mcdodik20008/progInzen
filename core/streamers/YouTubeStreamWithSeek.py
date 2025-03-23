import subprocess
import cv2
import numpy as np
from typing import Generator
import tempfile
import os
import yt_dlp

from core.video_stream import IVideoStreamSource


class YouTubeStreamWithSeek(IVideoStreamSource):
    """
    Потоковое чтение YouTube-видео с заданного времени без скачивания.
    """
    def __init__(self, youtube_url: str, start_time_sec: int = 0):
        self.youtube_url = youtube_url
        self.start_time_sec = start_time_sec
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        self.process = None
        self.cap = None

    def _get_direct_url(self) -> str:
        ydl_opts = {
            'quiet': True,
            'format': 'best',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(self.youtube_url, download=False)
            return info["url"]

    def start(self):
        direct_url = self._get_direct_url()
        start_time = self.start_time_sec

        command = [
            "ffmpeg",
            "-ss", str(start_time),           # стартовое время
            "-i", direct_url,                 # вход — прямой URL
            "-loglevel", "quiet",
            "-f", "mp4",                      # формат вывода
            "-movflags", "frag_keyframe+empty_moov",
            self.temp_file
        ]

        # Асинхронно запускаем ffmpeg, который перехватывает поток в .mp4-файл
        self.process = subprocess.Popen(command)

        # Подключаем OpenCV к этому файлу
        self.cap = cv2.VideoCapture(self.temp_file)

    def frames(self) -> Generator[np.ndarray, None, None]:
        if self.cap is None:
            raise RuntimeError("Stream not started")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def stop(self):
        if self.cap:
            self.cap.release()
        if self.process:
            self.process.terminate()
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
