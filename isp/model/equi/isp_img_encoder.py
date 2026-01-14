import torch
import torch.nn as nn


class BottleNeck(nn.Module):

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        down_sample=None,
        dilation=1,
        kernel_size=3,
    ):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=1, dilation=dilation, padding=0
        )

        padding = (kernel_size - 1) // 2
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding,
        )
        self.conv3 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            dilation=dilation,
            padding=0,
        )
        self.relu = nn.ReLU(inplace=True)

        self.down_sample = down_sample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.down_sample is not None:
            identity = self.down_sample(identity)
        out += identity
        out = self.relu(out)

        return out


class ResNet_50(nn.Module):
    def __init__(self, block, layers, obs_channel: int = 3, n_out: int = 512):
        super(ResNet_50, self).__init__()
        self.obs_channel = obs_channel
        self.in_planes = n_out // 16

        # 76x76
        self.conv1 = nn.Conv2d(
            obs_channel, self.in_planes, kernel_size=5, stride=1, padding=0
        )

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(
            block,
            n_out // 16,
            layers[0],
            stride=1,
        )
        self.layer2 = self._make_layer(
            block, n_out // 8, layers[1], stride=1, pool=True
        )
        self.layer3 = self._make_layer(
            block, n_out // 4, layers[2], stride=1, pool=True
        )
        self.layer4 = self._make_layer(block, n_out, layers[3], stride=1)

        self.out_layer = nn.Conv2d(n_out, n_out, kernel_size=3, padding=0)
        self.relu_out = nn.ReLU(inplace=True)

    def _make_layer(self, block, out_planes, block_num, stride=1, pool=False):
        down_sample = None
        if self.in_planes != out_planes:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_planes, kernel_size=1, stride=stride),
            )
        if pool:
            layers = [nn.MaxPool2d(kernel_size=2, padding=0, ceil_mode=True)]
        else:
            layers = []

        layers.append(
            block(self.in_planes, out_planes, stride=1, down_sample=down_sample)
        )

        self.in_planes = out_planes

        for i in range(1, block_num):
            layers.append(block(self.in_planes, out_planes, stride=1, down_sample=None))

        return nn.Sequential(*layers)

    def forward(self, x):

        #  76x76
        x = self.conv1(x)
        x = self.relu(x)
        # 72x72
        x = self.maxpool(x)
        # 36x36
        x = self.layer1(x)
        # 36x36
        x = self.layer2(x)
        # 18x18
        x = self.layer3(x)
        # 9x9
        x = self.layer4(x)
        out = self.out_layer(x)
        # 7x7
        return out


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        down_sample=None,
        dilation=1,
        kernel_size=3,
    ):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(kernel_size - 1) // 2,
        )

        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            padding=(kernel_size - 1) // 2,
        )

        self.relu = nn.ReLU(inplace=True)

        self.down_sample = down_sample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.down_sample is not None:
            identity = self.down_sample(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet_18(torch.nn.Module):
    def __init__(self, block, layers, obs_channel: int = 3, n_out: int = 512):
        super(ResNet_18, self).__init__()
        self.obs_channel = obs_channel
        self.in_planes = 64
        self.conv1 = nn.Conv2d(
            obs_channel, self.in_planes, kernel_size=7, stride=1, padding=3
        )

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, pool=True)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2, pool=True)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)

        self.out_layer = nn.Conv2d(256, n_out, kernel_size=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def _make_layer(self, block, out_planes, block_num, stride=1, pool=False):
        down_sample = None
        if self.in_planes != out_planes or stride != 1:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_planes, kernel_size=1, stride=stride),
            )
        layers = []
        if pool:
            layers.append(
                block(self.in_planes, out_planes, stride=2, down_sample=down_sample)
            )
        else:
            layers.append(
                block(self.in_planes, out_planes, stride=1, down_sample=down_sample)
            )

        self.in_planes = out_planes

        for i in range(1, block_num):
            layers.append(block(self.in_planes, out_planes, stride=1, down_sample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 76x76
        x = self.relu(self.conv1(x))
        # 72x72
        x = self.maxpool(x)
        # 36x36
        x = self.layer1(x)
        x = self.layer2(x)
        # 18x18
        x = self.layer3(x)
        # 9x9
        x = self.layer4(x)
        out = self.out_layer(x)
        out = self.avgpool(out)
        # 7x7
        return out


def Resnet18(obs_channel=3, out_fdim=128, **kwargs):
    model = ResNet_18(BasicBlock, [2, 2, 2, 2], obs_channel=obs_channel, n_out=out_fdim)

    return model


def Resnet34(obs_channel=3, out_fdim=128, **kwargs):
    model = ResNet_18(BasicBlock, [3, 4, 6, 3], obs_channel=obs_channel, n_out=out_fdim)

    return model


def Resnet50(obs_channel=3, out_fdim=128, **kwargs):
    model = ResNet_50(BottleNeck, [3, 4, 6, 3], obs_channel=obs_channel, n_out=out_fdim)

    return model
