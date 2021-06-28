from efficientnet_pytorch import EfficientNet
import torch.nn as nn

class EfficientNetFc(nn.Module):
    def __init__(self, backbone = 'efficientnet-b4', feature_dim=1792, projector_dim=128):
        super(EfficientNetFc, self).__init__()
        print(backbone)
        self.backbone = EfficientNet.from_pretrained(backbone)
        self.fc = nn.Linear(feature_dim, projector_dim)

    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = self.backbone._avg_pooling(x)
        if self.backbone._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self.backbone._dropout(x)
            y = self.fc(x)
        return y, x