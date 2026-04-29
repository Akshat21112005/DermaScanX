import torch.nn as nn
from transformers import SwinModel


class SwinClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = SwinModel.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224"
        )

        in_features = self.backbone.config.hidden_size

        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),   # head.0
            nn.Dropout(0.3),               # head.1
            nn.Linear(in_features, 512),   # head.2
            nn.GELU(),                     # head.3
            nn.BatchNorm1d(512),           # head.4
            nn.Dropout(0.2),               # head.5
            nn.Linear(512, num_classes)    # head.6
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        x = outputs.pooler_output
        return self.head(x)