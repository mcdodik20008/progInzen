import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel, ViTConfig


class ViViTClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, backbone="vit", dropout_rate=0.5):
        super(ViViTClassifier, self).__init__()

        self.backbone_type = backbone

        if backbone == "vit":
            if pretrained:
                self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
                hidden_size = self.backbone.config.hidden_size  # 768
            else:
                config = ViTConfig(
                    image_size=224,
                    patch_size=16,
                    num_channels=3,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    initializer_range=0.02,
                )
                self.backbone = ViTModel(config)
                hidden_size = config.hidden_size

            # Многоуровневый классификатор с дропаутом
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.LayerNorm(hidden_size // 2),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.GELU(),
                nn.LayerNorm(hidden_size // 4),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(hidden_size // 4, num_classes)
            )

            self.forward_func = self._forward_vit

        elif backbone == "resnet":
            # Используем ResNet50 как альтернативу
            self.backbone = models.resnet50(weights='DEFAULT')
            hidden_size = self.backbone.fc.in_features  # 2048
            self.backbone.fc = nn.Identity()

            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size // 2),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size // 4),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(hidden_size // 4, num_classes)
            )

            self.forward_func = self._forward_cnn

        elif backbone == "efficientnet":
            # EfficientNet для сравнения производительности
            self.backbone = models.efficientnet_b0(weights='DEFAULT')
            hidden_size = self.backbone.classifier[1].in_features  # 1280
            self.backbone.classifier = nn.Identity()

            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.SiLU(),
                nn.BatchNorm1d(hidden_size // 2),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 2, num_classes)
            )

            self.forward_func = self._forward_cnn

    def _forward_vit(self, x):
        outputs = self.backbone(x)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        return logits

    def _forward_cnn(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def forward(self, x):
        return self.forward_func(x)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_layers(self, n_layers=3):
        """Размораживает только последние n слоев основы модели"""
        if self.backbone_type == "vit":
            # Для ViT размораживаем последние n блоков Transformer
            for i, layer in enumerate(self.backbone.encoder.layer):
                requires_grad = i >= len(self.backbone.encoder.layer) - n_layers
                for param in layer.parameters():
                    param.requires_grad = requires_grad

        elif self.backbone_type in ["resnet", "efficientnet"]:
            # Для CNN размораживаем последние n слоев/блоков
            layers = list(self.backbone.children())
            for layer in layers[:-n_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            for layer in layers[-n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Для Apple M1/M2
    else:
        return torch.device("cpu")