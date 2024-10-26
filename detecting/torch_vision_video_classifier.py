from typing import List, Tuple

import numpy as np
import torch

from ultralytics.utils.torch_utils import select_device


class TorchVisionVideoClassifier:
    """Classifies videos using pretrained TorchVision models; see https://pytorch.org/vision/stable/."""

    from torchvision.models.video import (
        MViT_V1_B_Weights,
        MViT_V2_S_Weights,
        R3D_18_Weights,
        S3D_Weights,
        Swin3D_B_Weights,
        Swin3D_T_Weights,
        mvit_v1_b,
        mvit_v2_s,
        r3d_18,
        s3d,
        swin3d_b,
        swin3d_t,
    )

    model_name_to_model_and_weights = {
        "s3d": (s3d, S3D_Weights.DEFAULT),
        "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
        "swin3d_t": (swin3d_t, Swin3D_T_Weights.DEFAULT),
        "swin3d_b": (swin3d_b, Swin3D_B_Weights.DEFAULT),
        "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights.DEFAULT),
        "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights.DEFAULT),
    }

    def __init__(self, model_name: str, device: str or torch.device = ""):
        """
        Initialize the VideoClassifier with the specified model name and device.

        Args:
            model_name (str): The name of the model to use.
            device (str or torch.device, optional): The device to run the model on. Defaults to "".

        Raises:
            ValueError: If an invalid model name is provided.
        """
        if model_name not in self.model_name_to_model_and_weights:
            raise ValueError(f"Invalid model name '{model_name}'. Available models: {self.available_model_names()}")
        model, self.weights = self.model_name_to_model_and_weights[model_name]
        self.device = select_device(device)
        self.model = model(weights=self.weights).to(self.device).eval()

    @staticmethod
    def available_model_names() -> List[str]:
        """
        Get the list of available model names.

        Returns:
            list: List of available model names.
        """
        return list(TorchVisionVideoClassifier.model_name_to_model_and_weights.keys())

    def preprocess_crops_for_video_cls(self, crops: List[np.ndarray], input_size: list = None) -> torch.Tensor:
        """
        Preprocess a list of crops for video classification.

        Args:
            crops (List[np.ndarray]): List of crops to preprocess. Each crop should have dimensions (H, W, C)
            input_size (tuple, optional): The target input size for the model. Defaults to (224, 224).

        Returns:
            torch.Tensor: Preprocessed crops as a tensor with dimensions (1, T, C, H, W).
        """
        if input_size is None:
            input_size = [224, 224]
        from torchvision.transforms import v2

        transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(input_size, antialias=True),
                v2.Normalize(mean=self.weights.transforms().mean, std=self.weights.transforms().std),
            ]
        )

        processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]
        return torch.stack(processed_crops).unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)

    def __call__(self, sequences: torch.Tensor):
        """
        Perform inference on the given sequences.

        Args:
            sequences (torch.Tensor): The input sequences for the model. The expected input dimensions are
                                      (B, T, C, H, W) for batched video frames or (T, C, H, W) for single video frames.

        Returns:
            torch.Tensor: The model's output.
        """
        with torch.inference_mode():
            return self.model(sequences)

    def postprocess(self, outputs: torch.Tensor) -> Tuple[List[str], List[float]]:
        """
        Postprocess the model's batch output.

        Args:
            outputs (torch.Tensor): The model's output.

        Returns:
            List[str]: The predicted labels.
            List[float]: The predicted confidences.
        """
        pred_labels = []
        pred_confs = []
        for output in outputs:
            pred_class = output.argmax(0).item()
            pred_label = self.weights.meta["categories"][pred_class]
            pred_labels.append(pred_label)
            pred_conf = output.softmax(0)[pred_class].item()
            pred_confs.append(pred_conf)

        return pred_labels, pred_confs

    def freeze_layers(self, num_layers_to_freeze: int):
        """
        Freeze the first 'num_layers_to_freeze' layers of the model.

        Args:
            num_layers_to_freeze (int): Number of initial layers to freeze.
        """
        child_layers = list(self.model.children())
        for layer in child_layers[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"Froze first {num_layers_to_freeze} layers.")

    def prepare_to_fit(self):
            # Заморозка параметров модели, кроме последнего слоя
        for param in self.model.parameters():
            param.requires_grad = False

        # Обновление последнего слоя
        for param in self.model.visual_projection.parameters():
            param.requires_grad = True  # Разморозить новый слой
        return self.model