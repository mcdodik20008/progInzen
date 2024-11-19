import cv2

from controldetecting.BehaviorClassifier import BehaviorClassifier
from controldetecting.CapProcessor import CapProcessor
from controldetecting.PersonDetectorYOLOv11 import PersonDetectorYOLOv11


class VideoAnalyzer:
    def __init__(self, detector: PersonDetectorYOLOv11, classifier: BehaviorClassifier, cap_processor: CapProcessor):
        self.detector = detector
        self.behavior_classifier = classifier
        self.cap_processor = cap_processor

    def analyze_video(self, video_path, output_path, save_video=False):
        out = None
        save_to_disk = output_path is not None and save_video
        cap = cv2.VideoCapture(video_path)
        frame_width, frame_height, fps = self.cap_processor.get_video_properties(cap)

        if save_to_disk:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


        frame_counter = 0

        paused = False
        while cap.isOpened():
            if paused:
                self.set_pause()
                paused = False

            success, frame = cap.read()
            if not success:
                print("not success")
                break

            frame_counter += 1

            boxes, track_ids = self.detector(frame)
            if track_ids is not None:
                self.behavior_classifier.try_reset_to_infer(frame_counter)
                for box, track_id in zip(boxes, track_ids):
                    self.behavior_classifier.classify_behavior(box, track_id, frame, frame_counter)

                self.behavior_classifier.annotate_frame(boxes, frame)

            if save_to_disk:
                out.write(frame)
            cv2.imshow("YOLOv8 Tracking with S3D Classification", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                paused = True
            if key == ord("q"):
                break


    @staticmethod
    def set_pause():
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break