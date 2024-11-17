import cv2
import numpy as np
import torch
from torchvision import transforms

from transformers import AutoProcessor
from ultralytics.utils.plotting import Annotator


class CapProcessor:
    def __init__(self, model_name, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)

    @staticmethod
    def get_video_properties(cap):
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

    @staticmethod
    def crop_and_pad(frame, box, margin_percent=10):
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1

        # Add margin
        margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
        x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
        x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)

        # Take square crop from frame
        size = max(y2 - y1, x2 - x1)
        center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
        half_size = size // 2
        square_crop = frame[
                      max(0, center_y - half_size): min(frame.shape[0], center_y + half_size),
                      max(0, center_x - half_size): min(frame.shape[1], center_x + half_size),
                      ]

        return cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_LINEAR)

    def preprocess_crops_for_video_cls(self, crops: np.ndarray, fp16=False,
                                       input_size: list = None) -> torch.Tensor | None:
        if input_size is None:
            input_size = [224, 224]

        transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.float() / 255.0),
                transforms.Resize(input_size),
                transforms.Normalize(
                    mean=self.processor.image_processor.image_mean, std=self.processor.image_processor.image_std
                ),
            ]
        )

        processed_crops = []
        for crop in crops:
            tensor_crop = torch.from_numpy(crop).permute(2, 0, 1)  # (C, H, W)
            transformed_crop = transform(tensor_crop)  # Применяем трансформации
            processed_crops.append(transformed_crop)
        output = torch.stack(processed_crops).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        if fp16:
            output = output.half()

        return output

    @staticmethod
    def annotate_frame(frame, zipped_data):
        annotator = Annotator(frame, line_width=3, font_size=10, pil=False)
        for box, track_id, pred_label, pred_conf in zipped_data:
            top2_preds = sorted(zip(pred_label, pred_conf), key=lambda x: x[1], reverse=True)
            label_text = " | ".join([f"{label} ({conf:.2f})" for label, conf in top2_preds])
            annotator.box_label(box, label_text, color=(0, 0, 255))