import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, tubelet_size=(2, 16, 16)):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.proj(x)  # [B, embed_dim, T', H', W']
        x = x.flatten(2)  # [B, embed_dim, N]
        x = x.transpose(1, 2)  # [B, N, embed_dim]
        return x


class ViViT(nn.Module):
    def __init__(self,
                 image_size=224,
                 frames=16,
                 patch_size=(2, 16, 16),
                 in_channels=3,
                 num_classes=3,
                 embed_dim=768,
                 depth=8,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            tubelet_size=patch_size
        )

        num_patches = (frames // patch_size[0]) * \
                      (image_size // patch_size[1]) * \
                      (image_size // patch_size[2])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        # x: [B, T, H, W, C]
        x = x.permute(0, 4, 1, 2, 3)         # [B, C, T, H, W]
        x = self.patch_embed(x)              # [B, N, D]

        B, N, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, N+1, D]
        x = x + self.pos_embed[:, :N + 1, :]
        x = self.dropout(x)

        x = self.transformer(x)  # [B, N+1, D]
        cls_output = x[:, 0]    # [B, D]
        return self.head(cls_output)  # [B, num_classes]