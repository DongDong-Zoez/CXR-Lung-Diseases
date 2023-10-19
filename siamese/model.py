import timm
import torch
import torch.nn as nn

class Siamese(nn.Module):

    def __init__(self, num_classes=3, model_name="tv_densenet121", pretrained=True):
        super().__init__()

        self.features = timm.create_model(model_name, num_classes=0, pretrained=pretrained)
        self.classifier = nn.Linear(self.__get_in_channels()*2, num_classes)

    def __get_in_channels(self):
        x = torch.randn(1,3,224,224)

        return self.features(x).shape[-1]

    def forward(self, before_image, after_image):
        before_features = self.features(before_image)
        after_features = self.features(after_image)
        fusion_features = torch.cat([before_features, after_features], dim=1)
        out = self.classifier(fusion_features)
        return out