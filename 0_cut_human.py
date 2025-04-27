import os

from core.LiveAnalyzer import LiveAnalyzer
from core.controller import AppController
from core.frame_processor import FrameProcessor
from core.streamers.SeekableVideoFileSource import SeekableVideoFileSource
from core.track_clip_buffer import TrackClipBuffer


def find_all_videos(folder, extensions=(".mp4", ".avi", ".mov")):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(extensions):
                yield os.path.join(root, file)

def process_video(video_path, output_dir):
    print(f"[INFO] Обработка видео: {video_path}")
    controller = AppController(
        stream = SeekableVideoFileSource(video_path),
        clip_buffer=TrackClipBuffer(output_dir),
        processor=FrameProcessor()
    )
    controller.run()

def main():
    VIDEO_DIR = r"./dataset_Real Life Violence Dataset"  # или путь к общей папке
    OUTPUT_PATH = r"./clips"  # если у тебя TrackClipBuffer используется

    for video_file in find_all_videos(VIDEO_DIR):
        try:
            process_video(video_file, OUTPUT_PATH)
        except Exception as e:
            print(f"[ERROR] Ошибка при обработке {video_file}: {e}")

if __name__ == "__main__":
    main()
