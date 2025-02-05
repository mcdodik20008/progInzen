import cv2

from CapProcessor import CapProcessor
from CarAccidentClassifier import CarAccidentClassifier
from FrameAnnotator import FrameAnnotator
from PersonDetectorYOLOv11 import PersonDetectorYOLOv11
from controldetecting.analyzer.Classifier import Classifier
from controldetecting.model.AnnotateData import AnnotateData


class VideoAnalyzer:
    def __init__(self, detector: PersonDetectorYOLOv11, classifier: Classifier, car_accident: CarAccidentClassifier,
                 cap_processor: CapProcessor, frame_annotator: FrameAnnotator):
        self.detector = detector
        self.behavior_classifier = classifier
        self.cap_processor = cap_processor
        self.car_accident = car_accident
        self.frame_annotator = frame_annotator

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

            human_boxes, car_boxes = [], []
            human_classes = []

            frame_counter += 1
            frame = self.cap_processor.resize_frame(cap, frame)
            boxes, track_ids, classes = self.detector(frame)
            if track_ids is not None:
                self.behavior_classifier.try_reset_to_predict(frame_counter)
                for box, track_id, class_id in zip(boxes, track_ids, classes):
                    if class_id == 0:
                        self.behavior_classifier(frame, box, track_id, frame_counter)
                        human_boxes.append(box)
                        human_classes.append(class_id)
                    elif class_id == 2:
                        self.car_accident(frame, box, class_id)
                        car_boxes.append(box)

            human_data = self.behavior_classifier.get_data_for_annotate()
            self.frame_annotator(frame, human_boxes, human_data)

            car_data = AnnotateData.get_template(len(car_boxes), "car", "blue")
            self.frame_annotator(frame, car_boxes, car_data)

            if save_to_disk:
                out.write(frame)
            cv2.imshow("Angry Birds", frame)

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
