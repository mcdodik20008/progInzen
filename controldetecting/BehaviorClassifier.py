import torch
from transformers import AutoModel, AutoProcessor
from typing import List, Tuple
from torchvision import transforms

from controldetecting.CapProcessor import CapProcessor


class BehaviorClassifier:
    def __init__(self, cap_processor: CapProcessor, device, model_name="microsoft/xclip-base-patch32", ):
        self.device = device
        self.cap_processor = cap_processor

        # Загружаем модель и токенайзер для классификации поведения
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Настройки трансформаций для входного изображения
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.labels = ["Aggressive", "Normal Behavior"]

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
