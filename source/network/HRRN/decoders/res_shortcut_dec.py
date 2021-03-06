from network.HRRN.decoders.resnet_dec import ResNet_D_Dec

class ResShortCut_D_Dec(ResNet_D_Dec):

    def __init__(self, block, layers, norm_layer=None, large_kernel=False, late_downsample=False):
        super(ResShortCut_D_Dec, self).__init__(block,layers,norm_layer,large_kernel,late_downsample=late_downsample)

    def forward(self, x, mid_fea):
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        x = self.layer3(x) + fea3
        x = self.layer4(x) + fea2
        x = self.conv1(x)
        x = self.bn1(x)
        decoder = self.leaky_relu(x) + fea1
        x = self.conv2(decoder)
        high_sal = (self.tanh(x) + 1.0) / 2.0
        return high_sal

