from core.frame_processor import FrameProcessor
from core.track_clip_buffer import TrackClipBuffer
from core.video_stream import IVideoStreamSource


class AppController:
    """
    Управляет основным циклом обработки видеопотока:
    - захват кадров,
    - обнаружение людей,
    - накопление и сохранение клипов.
    """
    def __init__(
        self,
        stream : IVideoStreamSource,
        clip_buffer: TrackClipBuffer,
        processor: FrameProcessor,
    ):
        self.stream = stream
        self.processor = processor
        self.clip_buffer = clip_buffer

    def run(self):
        print("[AppController] Запуск видеопотока...")
        self.stream.start()

        try:
            for frame in self.stream.frames():
                detections = self.processor.detect(frame)

                for i, det in enumerate(detections):
                    if not det.is_person():
                        continue

                    x1, y1, x2, y2 = map(int, det.bbox)
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size > 0:
                        self.clip_buffer.add(det.track_id, cropped)

        except KeyboardInterrupt:
            print("[AppController] Завершение по Ctrl+C")

        finally:
            self.stream.stop()
