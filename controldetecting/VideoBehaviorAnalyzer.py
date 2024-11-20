import cv2

from BehaviorClassifier import BehaviorClassifier
from CapProcessor import CapProcessor
from CarAccidentClassifier import CarAccidentClassifier
from PersonDetectorYOLOv11 import PersonDetectorYOLOv11


class VideoAnalyzer:
    def __init__(self, detector: PersonDetectorYOLOv11, classifier: BehaviorClassifier, car_accident: CarAccidentClassifier, cap_processor: CapProcessor):
        self.detector = detector
        self.behavior_classifier = classifier
        self.cap_processor = cap_processor
        self.car_accident = car_accident

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
            boxes, track_ids, classes = self.detector(frame)
            if track_ids is not None:
                self.behavior_classifier.try_reset_to_infer(frame_counter)
                for box, track_id, class_id in zip(boxes, track_ids, classes):
                    if class_id == 0:
                        print(f"class_id = {class_id}")
                        self.behavior_classifier.invoke(frame, box, track_id, frame_counter)
                    # if class_id == 2:
                    #     self.car_accident(frame, box, class_id)

                self.behavior_classifier.annotate_frame(boxes, frame, classes)

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