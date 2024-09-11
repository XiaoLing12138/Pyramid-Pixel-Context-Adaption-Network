import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class Points(nn.Module):
    def forward(self, x):
        y = x.transpose(1, 2)
        y_mean = torch.mean(y, dim=2)
        y_mean = torch.unsqueeze(y_mean, dim=1)
        y_std = torch.std(y, unbiased=False, dim=2)
        y_std = torch.unsqueeze(y_std, dim=1)
        x = (x - y_mean) / y_std
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.mean(x, 1).unsqueeze(1)


class PPCALayer(nn.Module):
    def __init__(self, gate_channel, layer_idx, groups=4):
        super(PPCALayer, self).__init__()

        num_fea = [56 ** 2, 28 ** 2, 14 ** 2, 7 ** 2][layer_idx]

        self.channels = gate_channel
        self.groups = groups
        self.compress = ChannelPool()
        self.points = Points()

        self.scale = [1, 2, 4, 8, 16, 32, 64, 128]
        self.weight = Parameter(torch.Tensor(num_fea, sum(self.scale[:groups])))
        self.weight.data.fill_(0)
        self.bias = Parameter(torch.Tensor(num_fea))

        self.sigmoid = nn.Sigmoid()

    def _style_integration(self, t):
        z = t * self.weight[None, :, :]  # B x C x 3
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1
        return z

    def forward(self, x):
        b, c, height, width = x.size()

        feature_list = []
        for i in self.scale[:self.groups]:
            for j in range(i):
                feature_list.append(
                    self.compress(x[:, self.channels * j // i:self.channels * (j + 1) // i, :, :]).view(b, 1,
                                                                                                        height * width))

        y_t = torch.cat(feature_list, 1)

        y_t = self.points(y_t)
        y = y_t.transpose(1, 2)

        out = self._style_integration(y)

        out = out.transpose(1, 2)
        out = self.sigmoid(out.view(b, 1, height, width))
        # broadcasting
        return x * out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, layer_idx, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.ppca = PPCALayer(planes, layer_idx)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.ppca(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, layer_idx, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.ppca = PPCALayer(planes * 4, layer_idx)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.ppca(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, network_type, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, 0, layers[0])
        self.layer2 = self._make_layer(block, 128, 1, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, 2, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, 3, layers[3], stride=2)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, layer_index, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, layer_index, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, layer_index))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResidualNet(network_type, depth, num_classes):
    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes)

    return model


def ppca_resnet18(num_classes=3):
    model = ResidualNet("ImageNet", 18, num_classes)
    return model


def ppca_resnet34(num_classes=3):
    model = ResidualNet("ImageNet", 34, num_classes)
    return model


def ppca_resnet50(num_classes=3):
    model = ResidualNet("ImageNet", 50, num_classes)
    return model


def ppca_resnet101(num_classes=3):
    model = ResidualNet('ImageNet', 101, num_classes)
    return model


if __name__ == '__main__':
    temp = torch.randn((1, 3, 224, 224))

    net = ppca_resnet18()
    out = net(temp)

    print(out)
