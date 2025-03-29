import time

import torch
import torch.nn.functional as F
import numpy as np
from model.vivit import ViViT
import cv2

LABEL_MAP = {0: "aggressive", 1: "suspicious", 2: "normal"}


class BehaviorClassifier:
    def __init__(
            self,
            model_path: str = "ready_models/v1/model.pth",
            num_frames: int = 16,
            frame_size: tuple = (224, 224),
            device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViViT(num_classes=3)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.num_frames = num_frames
        self.frame_size = frame_size

    def __call__(self, frames: list[np.ndarray]) -> tuple[str, dict]:
        """
        frames — список кадров [np.ndarray (H, W, C)]
        """
        start_time = time.time()

        if len(frames) < self.num_frames:
            print(f"[⚠️] Кадров меньше {self.num_frames}, дублируем последние")
            frames += [frames[-1]] * (self.num_frames - len(frames))
        else:
            frames = frames[:self.num_frames]

        # Resize + normalize
        processed = [cv2.resize(f, self.frame_size).astype(np.float32) / 255.0 for f in frames]
        video = np.stack(processed)  # [T, H, W, C]

        x = torch.tensor(video).unsqueeze(0).to(self.device)  # [1, T, H, W, C]

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]
            class_id = torch.argmax(probs).item()

        elapsed = time.time() - start_time
        probs_dict = {
            LABEL_MAP[i]: round(probs[i].item(), 4)
            for i in range(len(LABEL_MAP))
        }

        # 📋 Логирование
        print(f"\n[🔍] BehaviorClassifier Inference:")
        print(f"   ▸ Размер входа: {video.shape} (T, H, W, C)")
        print(f"   ▸ Предсказание: {LABEL_MAP[class_id]}")
        print(f"   ▸ Вероятности: {probs_dict}")
        print(f"   ▸ Время инференса: {elapsed:.3f} сек\n")

        return LABEL_MAP[class_id], probs_dict

if __name__ == "__main__":
    BehaviorClassifier()