import torch.nn as nn
from   network.HRRN.ops import SpectralNorm

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = SpectralNorm(conv3x3(inplanes, planes, stride))
        self.bn1 = norm_layer(planes)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = SpectralNorm(conv3x3(planes, planes))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out

class ResNet_D(nn.Module):

    def __init__(self, block, layers, norm_layer=None, late_downsample=False):
        super(ResNet_D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.late_downsample = late_downsample
        self.midplanes = 64 if late_downsample else 32
        self.start_stride = [1, 2, 1, 2] if late_downsample else [2, 1, 2, 1]
        self.conv1 = SpectralNorm(nn.Conv2d(3 + 3, 32, kernel_size=3,
                                            stride=self.start_stride[0], padding=1, bias=False))
        self.conv2 = SpectralNorm(nn.Conv2d(32, self.midplanes, kernel_size=3, stride=self.start_stride[1], padding=1,
                                            bias=False))
        self.conv3 = SpectralNorm(nn.Conv2d(self.midplanes, self.inplanes, kernel_size=3, stride=self.start_stride[2],
                                            padding=1, bias=False))
        self.bn1 = norm_layer(32)
        self.bn2 = norm_layer(self.midplanes)
        self.bn3 = norm_layer(self.inplanes)
        self.activation = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=self.start_stride[3])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer_bottleneck = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight_bar)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
        self.conv1.module.weight_bar.data[:,3:,:,:] = 0

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks == 0:
            return nn.Sequential(nn.Identity())
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(2, stride),
                SpectralNorm(conv1x1(self.inplanes, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SpectralNorm(conv1x1(self.inplanes, planes * block.expansion, stride)),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)



