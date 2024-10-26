from typing import List, Tuple

import torch
from torchvision import transforms
import numpy as np
from transformers import AutoModel, AutoProcessor

from ultralytics.utils.torch_utils import select_device


class HuggingFaceVideoClassifier:
    """Zero-shot video classifier using Hugging Face models for various devices."""

    def __init__(
        self,
        labels: List[str],
        model_name: str = "microsoft/xclip-base-patch16-zero-shot",
        device: str or torch.device = "",
        fp16: bool = False,
    ):
        """
        Initialize the HuggingFaceVideoClassifier with the specified model name.

        Args:
            labels (List[str]): List of labels for zero-shot classification.
            model_name (str): The name of the model to use. Defaults to "microsoft/xclip-base-patch16-zero-shot".
            device (str or torch.device, optional): The device to run the model on. Defaults to "".
            fp16 (bool, optional): Whether to use FP16 for inference. Defaults to False.
        """
        self.fp16 = fp16
        self.labels = labels
        self.device = select_device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        if fp16:
            model = model.half()
        self.model = model.eval()

    def preprocess_crops_for_video_cls(self, crops: List[np.ndarray], input_size: list = None) -> torch.Tensor | None:
        """
        Preprocess a list of crops for video classification.

        Args:
            crops (List[np.ndarray]): List of crops to preprocess. Each crop should have dimensions (H, W, C)
            input_size (tuple, optional): The target input size for the model. Defaults to (224, 224).

        Returns:
            torch.Tensor: Preprocessed crops as a tensor (1, T, C, H, W).
        """
        if input_size is None:
            input_size = [224, 224]

        # Проверяем, что crops не пустой
        if not crops:
            print("Error: No crops provided for preprocessing!")
            return None

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
            # Проверяем, что форма каждого crop корректная
            if crop.ndim != 3 or crop.shape[2] != 3:
                print(f"Invalid crop shape: {crop.shape}")
                return None

            tensor_crop = torch.from_numpy(crop).permute(2, 0, 1)  # (C, H, W)
            transformed_crop = transform(tensor_crop)  # Применяем трансформации
            processed_crops.append(transformed_crop)

        # Логируем форму каждого обработанного crop
        for idx, crop in enumerate(processed_crops):
            print(f"Processed crop {idx} shape: {crop.shape}")  # Вывод формы

        output = torch.stack(processed_crops).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        if output is None:
            print("Error: Output is None after stacking processed crops!")
            return None

        if self.fp16:
            output = output.half()

        return output

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on the given sequences.

        Args:
            sequences (torch.Tensor): The input sequences for the model. Batched video frames with shape (B, T, H, W, C).

        Returns:
            torch.Tensor: The model's output.
        """
        input_ids = self.processor(text=self.labels, return_tensors="pt", padding=True)["input_ids"].to(self.device)

        inputs = {"pixel_values": sequences, "input_ids": input_ids}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        return outputs.logits_per_video

    def postprocess(self, outputs: torch.Tensor) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Postprocess the model's batch output.

        Args:
            outputs (torch.Tensor): The model's output.

        Returns:
            List[List[str]]: The predicted top3 labels.
            List[List[float]]: The predicted top3 confidences.
        """
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

    def prepare_to_fit(self):
            # Заморозка параметров модели, кроме последнего слоя
        for param in self.model.parameters():
            param.requires_grad = False

        # Обновление последнего слоя
        for param in self.model.visual_projection.parameters():
            param.requires_grad = True  # Разморозить новый слой
        return self.model
