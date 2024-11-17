import time
from collections import defaultdict

import cv2
import torch

from controldetecting.BehaviorClassifier import BehaviorClassifier
from controldetecting.CapProcessor import CapProcessor
from controldetecting.PersonDetectorYOLOv11 import PersonDetectorYOLOv11


class VideoAnalyzer:
    def __init__(self, detector: PersonDetectorYOLOv11, classifier: BehaviorClassifier, cap_processor: CapProcessor):
        self.detector = detector
        self.classifier = classifier
        self.cap_processor = cap_processor
        self.num_video_sequence_samples = 8
        self.video_cls_overlap_ratio: float = 0.25

    def analyze_video(self, video_path, output_path, save_video=False, skip_frame=2):
        out = None
        save_to_disk = output_path is not None and save_video
        cap = cv2.VideoCapture(video_path)
        frame_width, frame_height, fps = self.cap_processor.get_video_properties(cap)

        if save_to_disk:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        track_history = defaultdict(list)
        frame_counter = 0

        track_ids_to_infer, crops_to_infer = [], []
        pred_labels, pred_confs = [], []
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
                # сбрасываем каждый цикл то, что будем выводить
                if frame_counter % skip_frame == 0:
                    crops_to_infer = []
                    track_ids_to_infer = []

                for box, track_id in zip(boxes, track_ids):
                    track_by_id = track_history[track_id]
                    frame_mod_skip = frame_counter % skip_frame

                    can_add = self.can_add_track_and_crop_to_infer(box, frame, frame_mod_skip, track_by_id)
                    if can_add:
                        corps = self.process_frame(track_by_id)
                        crops_to_infer.append(corps)
                        track_ids_to_infer.append(track_id)

                if self.can_classify_behavior(crops_to_infer, frame_counter, pred_labels, skip_frame):
                    crops_batch = torch.cat(crops_to_infer, dim=0)
                    start_inference_time = time.time()
                    output_batch = self.classifier(crops_batch)
                    end_inference_time = time.time()
                    inference_time = end_inference_time - start_inference_time
                    print(f"video cls inference time: {inference_time:.4f} seconds")
                    pred_labels, pred_confs = self.classifier.postprocess(output_batch)

                if track_ids_to_infer and crops_to_infer:
                    zipped_data = zip(boxes, track_ids_to_infer, pred_labels, pred_confs)
                    self.cap_processor.annotate_frame(frame, zipped_data)

            if save_to_disk:
                out.write(frame)
            cv2.imshow("YOLOv8 Tracking with S3D Classification", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                paused = True
            if key == ord("q"):
                break

    def can_add_track_and_crop_to_infer(self, box, frame, frame_mod_skip, track_by_id):
        if frame_mod_skip == 0:
            crop = self.cap_processor.crop_and_pad(frame, box)
            track_by_id.append(crop)

        # если мы накопили больше кадров, чем надо (8), то выкинем, что бы не переполнить
        if len(track_by_id) > self.num_video_sequence_samples:
            track_by_id.pop(0)
        # если накопили нужное количество кадров по выбранному объекту, то можно препроцессить кропс и ставить в очередь на классификацию
        return len(track_by_id) == self.num_video_sequence_samples and frame_mod_skip == 0

    def can_classify_behavior(self, crops_to_infer, frame_counter, pred_labels, skip_frame):
        return crops_to_infer and (
                not pred_labels
                or frame_counter % int(
            self.num_video_sequence_samples * skip_frame * (1 - self.video_cls_overlap_ratio)) == 0
        )

    def process_frame(self, track_by_id):
        start_time = time.time()
        crops = self.cap_processor.preprocess_crops_for_video_cls(track_by_id)
        end_time = time.time()
        preprocess_time = end_time - start_time
        print(f"video cls preprocess time: {preprocess_time:.4f} seconds")
        return crops


    @staticmethod
    def set_pause():
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break