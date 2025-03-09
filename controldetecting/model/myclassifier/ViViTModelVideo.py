# ViViTModelVideo.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig


class Attention3D(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention3D, self).__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, T, D]
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class ViViTVideoClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, backbone="vit", dropout_rate=0.5):
        super(ViViTVideoClassifier, self).__init__()

        self.backbone_type = backbone
        self.frame_feature_extractor = ViViTClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            backbone=backbone,
            dropout_rate=dropout_rate
        )

        # Удаляем классификатор из экстрактора признаков
        if backbone == "vit":
            hidden_size = self.frame_feature_extractor.backbone.config.hidden_size
            self.get_features = self._get_vit_features
        elif backbone == "resnet":
            hidden_size = 2048
            self.get_features = self._get_cnn_features
        elif backbone == "efficientnet":
            hidden_size = 1280
            self.get_features = self._get_cnn_features

        # Временная обработка с механизмом внимания
        self.temporal_attention = Attention3D(hidden_size)

        # Временное объединение
        self.temporal_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1)
        )

        # Классификатор для последовательностей
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def _get_vit_features(self, x):
        outputs = self.frame_feature_extractor.backbone(x)
        return outputs.last_hidden_state[:, 0]  # только CLS-токен

    def _get_cnn_features(self, x):
        return self.frame_feature_extractor.backbone(x)

    def forward(self, x):
        # x: [B, T, C, H, W] где B - батч, T - число кадров
        batch_size, num_frames = x.size(0), x.size(1)

        # Преобразуем в [B*T, C, H, W] для обработки каждого кадра отдельно
        x_reshaped = x.view(-1, x.size(2), x.size(3), x.size(4))

        # Получаем признаки для каждого кадра
        frame_features = self.get_features(x_reshaped)  # [B*T, D]

        # Восстанавливаем временное измерение [B, T, D]
        temporal_features = frame_features.view(batch_size, num_frames, -1)

        # Обрабатываем временную информацию с помощью механизма внимания
        temporal_features = self.temporal_attention(temporal_features)

        # Объединяем временную информацию
        pooled_features = self.temporal_pool(temporal_features.transpose(1, 2)).squeeze(-1)

        # Классифицируем
        output = self.classifier(pooled_features)

        return output

    def freeze_backbone(self):
        for param in self.frame_feature_extractor.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.frame_feature_extractor.backbone.parameters():
            param.requires_grad = True

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Для Apple M1/M2
    else:
        return torch.device("cpu")