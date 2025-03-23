from core.LiveAnalyzer import LiveAnalyzer
from core.controller import AppController
from core.frame_processor import FrameProcessor
from core.track_clip_buffer import TrackClipBuffer
from core.streamers.SeekableVideoFileSource import SeekableVideoFileSource


def main():
    VIDEO_PATH = r"C:\obsrecords\2025-03-23 11-42-35.mp4"
    OUTPUT_PATH = r"./clips"
    controller = AppController(
        stream = SeekableVideoFileSource(VIDEO_PATH),
        clip_buffer=TrackClipBuffer(OUTPUT_PATH),
        processor=FrameProcessor()
    )

    # controller.run()
    stream = SeekableVideoFileSource(VIDEO_PATH, start_time_sec=30)
    analyzer = LiveAnalyzer(stream)
    analyzer.run()

if __name__ == "__main__":
    main()
