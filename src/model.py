import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights


class VGGFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        print("Loading pretrained VGG19 weights...")

        weights = VGG19_Weights.DEFAULT
        vgg = models.vgg19(weights=weights).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg
        self.layer_name_mapping = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",
            "28": "conv5_1",
        }

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layer_name_mapping:
                features[self.layer_name_mapping[name]] = x
        return features
