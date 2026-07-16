import torch.nn as nn
import torchvision


class TransferFCN(nn.Module):
    """FCN whose encoder is an ImageNet-pretrained ResNet34 (Q5 option b).
    [AI-assisted: Claude Code]


    The avgpool and fc layers of ResNet34 are removed so a 224x224 input
    yields a (512, 7, 7) feature map, which feeds the same five-deconv
    decoder used by the baseline FCN.
    """

    def __init__(self, n_class):
        # [AI-assisted: Claude Code]
        super().__init__()
        self.n_class = n_class
        resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        # everything except avgpool and fc: output (N, 512, H/32, W/32)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        # [AI-assisted: Claude Code]
        x5 = self.encoder(x)

        y1 = self.bn1(self.relu(self.deconv1(x5)))
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        y5 = self.bn5(self.relu(self.deconv5(y4)))

        score = self.classifier(y5)

        return score  # size=(N, n_class, H, W)
