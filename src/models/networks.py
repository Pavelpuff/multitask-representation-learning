import torch
import torch.nn as nn
import torch.nn.functional as F


class MultitaskModel(nn.Module):
    """
    Multitask neural network with a shared CNN backbone and two MLP heads:
    1. Classification (135 classes).
    2. Regression (6 classes for counting).
    """

    def __init__(self, dropout_rate: float = 0.4) -> None:
        super().__init__()

        # Fixed CNN Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
        )

        self.head_cls = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(256, 135))

        self.head_cnt = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(256, 6))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)

        cls_logits = self.head_cls(features)
        cnt_preds = self.head_cnt(features)

        # NLLLoss expects log probabilities
        log_probs = F.log_softmax(cls_logits, dim=1)

        return log_probs, cnt_preds
