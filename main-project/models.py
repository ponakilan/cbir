import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights
from torchvision import io


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        resnet_backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.resnet_backbone = nn.Sequential(*list(resnet_backbone.children())[:-2])

        self.gem = GeM(p=3)

    def forward(self, x):
        x = self.resnet_backbone(x)
        x = self.gem(x)
        return x


if __name__ == "__main__":
    extractor = FeatureExtractor()

    sample_image_path = "/home/ubuntu/data/datasets/roxford5k/jpg/oxford_002881.jpg"
    sample_image = io.read_image(sample_image_path).unsqueeze(0).float()
    print(sample_image.shape)
    features = extractor(sample_image)

    print(features.shape)
