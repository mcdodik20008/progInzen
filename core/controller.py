from core.frame_processor import FrameProcessor
from core.streamers.SeekableVideoFileSource import SeekableVideoFileSource
from core.track_clip_buffer import TrackClipBuffer
from core.video_stream import IVideoStreamSource

class AppController:
    def __init__(self, stream: IVideoStreamSource, clip_buffer: TrackClipBuffer, processor: FrameProcessor):
        self.stream = stream
        self.processor = processor
        self.clip_buffer = clip_buffer

    def run(self):
        print("[AppController] Запуск видеопотока...")
        self.stream.start()
        try:
            for frame in self.stream.frames():
                detections = self.processor.detect(frame)
                for det in detections:
                    if det.is_person():
                        x1, y1, x2, y2 = map(int, det.bbox)
                        cropped = frame[y1:y2, x1:x2]
                        if cropped.size > 0:
                            self.clip_buffer.add(det.track_id, cropped)
        except KeyboardInterrupt:
            print("[AppController] Завершение по Ctrl+C")
        finally:
            self.stream.stop()


def get_ui_meta():
    return {
        "id": "cut_human",
        "name": "✂️ Нарезка батчей по трекам",
        "description": "YOLO + трекинг: вырезает людей из видео и сохраняет кадры по track_id.",
        "order": 1,
        "parameters": {
            "VIDEO_PATH": {"type": "str", "default": "./input/video.mp4"},
            "OUTPUT_PATH": {"type": "str", "default": "./clips"}
        }
    }

def main(video_path: str, output_path: str):
    controller = AppController(
        stream=SeekableVideoFileSource(video_path),
        clip_buffer=TrackClipBuffer(output_path),
        processor=FrameProcessor()
    )
    controller.run()
