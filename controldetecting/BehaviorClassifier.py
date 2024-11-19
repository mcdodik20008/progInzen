import torch
from transformers import AutoModel, AutoProcessor
from typing import List, Tuple
from torchvision import transforms
from collections import defaultdict
from controldetecting.CapProcessor import CapProcessor
import time

class BehaviorClassifier:
    def __init__(self, cap_processor: CapProcessor, device, model_name="microsoft/xclip-base-patch32", ):
        self.device = device
        self.cap_processor = cap_processor

        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.labels = [
            "calm pedestrian", "social interaction", "street vendor", "public transportation user", "recreational activity",
            "physical altercation", "aggressive gestures", "property damage", "harassment",
            "loitering", "street performer", "protesting"
                       ]


        self.track_history = defaultdict(list)
        self.track_ids_to_infer, self.crops_to_infer = [], []
        self.pred_labels, self.pred_confs = [], []
        self.num_video_sequence_samples = 8
        self.video_cls_overlap_ratio: float = 0.25
        self.skip_frame = 2

        self.processed_box = []

    def try_reset_to_infer(self, frame_counter):
        if frame_counter % self.skip_frame == 0:
            self.crops_to_infer = []
            self.track_ids_to_infer = []

    def classify_behavior(self, box, track_id, frame, frame_counter):
        self.processed_box.append(box)
        track_by_id = self.track_history[track_id]
        frame_mod_skip = frame_counter % self.skip_frame

        can_add = self.can_add_track_and_crop_to_infer(box, frame, frame_mod_skip, track_by_id)
        if can_add:
            corps = self.process_frame(track_by_id)
            self.crops_to_infer.append(corps)
            self.track_ids_to_infer.append(track_id)

        if self.can_classify_behavior(self.crops_to_infer, frame_counter, self.pred_labels, self.skip_frame):
            crops_batch = torch.cat(self.crops_to_infer, dim=0)
            start_inference_time = time.time()
            output_batch = self(crops_batch)
            end_inference_time = time.time()
            inference_time = end_inference_time - start_inference_time
            print(f"video cls inference time: {inference_time:.4f} seconds")
            self.pred_labels, self.pred_confs = self.postprocess(output_batch)

    def annotate_frame(self,boxes, frame):
        if self.track_ids_to_infer and self.crops_to_infer:
            zipped_data = zip(boxes, self.pred_labels, self.pred_confs)
            self.cap_processor.annotate_frame(frame, zipped_data)
            self.processed_box.clear()

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        input_ids = self.processor(text=self.labels, return_tensors="pt", padding=True)["input_ids"].to(self.device)
        inputs = {"pixel_values": sequences, "input_ids": input_ids}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs.logits_per_video

    def postprocess(self, outputs: torch.Tensor) -> Tuple[List[List[str]], List[List[float]]]:
        pred_labels = []
        pred_confs = []
        with torch.no_grad():
            logits_per_video = outputs  # Assuming outputs is already the logits tensor
            probs = logits_per_video.softmax(dim=-1)  # Use softmax to convert logits to probabilities
        for prob in probs:
            top2_indices = prob.topk(2).indices.tolist()
            top2_labels = [self.labels[idx] for idx in top2_indices]
            top2_confs = prob[top2_indices].tolist()
            pred_labels.append(top2_labels)
            pred_confs.append(top2_confs)
        return pred_labels, pred_confs

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
            self.num_video_sequence_samples * skip_frame * (1 - self.video_cls_overlap_ratio)) == 0)

    def process_frame(self, track_by_id):
        start_time = time.time()
        crops = self.cap_processor.preprocess_crops_for_video_cls(track_by_id)
        end_time = time.time()
        preprocess_time = end_time - start_time
        print(f"video cls preprocess time: {preprocess_time:.4f} seconds")
        return crops