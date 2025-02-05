import cv2
import numpy as np
import torch
from torchvision import transforms

from transformers import AutoProcessor


class CapProcessor:
    def __init__(self, processor_name, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(processor_name)

    def __call__(self, crops: np.ndarray, fp16=False,
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
    def get_video_properties(cap):
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(
            cv2.CAP_PROP_FPS)

    @staticmethod
    def crop_and_pad(frame, box, resize = (224, 224), margin_percent=10):
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

        return cv2.resize(square_crop, resize, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def resize_frame(cap, frame):
        target_width = 1024

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = original_width / original_height
        target_height = int(target_width / aspect_ratio)

        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)