import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from itertools import chain
import torch.nn.functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)
batchnorm_momentum = 0.01 / 2


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _BNReluConv(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size=3, batch_norm=True,
                 bn_momentum=0.1, bias=False, dilation=1):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(channels_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = kernel_size // 2
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size, padding=padding, bias=bias, dilation=dilation))


class _Upsample(nn.Module):
    def __init__(self, channels_in, skip_maps_in, channels_out, use_bn=True, kernel_size=3):
        super(_Upsample, self).__init__()
        # print(f'Upsample layer: in = {channels_in}, skip = {skip_maps_in}, out = {channels_out}')
        self.bottleneck = _BNReluConv(skip_maps_in, channels_in, kernel_size=1, batch_norm=use_bn)
        self.blend_conv = _BNReluConv(channels_in, channels_out, kernel_size=kernel_size, batch_norm=use_bn)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        skip_size = skip.size()[2:4]
        x = upsample(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)
        return x


class SpatialPyramidPooling(nn.Module):

    def __init__(self, channels_in, num_levels, bt_size=512, level_size=128, channels_out=128,
                 grids=(8, 4, 2, 1), bn_momentum=0.1, use_bn=True):
        super(SpatialPyramidPooling, self).__init__()
        self.grids = grids
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn',
                            _BNReluConv(channels_in, bt_size, kernel_size=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNReluConv(num_features, level_size, kernel_size=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        self.spp.add_module('spp_fuse',
                            _BNReluConv(final_size, channels_out, kernel_size=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        # self.SPP_FUSE = _BNReluConv()

    def forward(self, x):
        levels = []
        target_size = x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):

            grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
            x_pooled = F.adaptive_avg_pool2d(x, grid_size)

            level = self.spp[i].forward(x_pooled)

            level = upsample(level, target_size)
            levels.append(level)
        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x


class MySpatialPyramidPooling(nn.Module):

    def __init__(self, channels_in, channels_out, level_num, spp_channels, level_channels,
                 grid=(8, 4, 2, 1), bn_momentum=0.1):
        super(MySpatialPyramidPooling, self).__init__()

        self.grid = grid
        self.level_num = level_num
        self.SPP_BN = _BNReluConv(channels_in, spp_channels,
                                  kernel_size=1, bn_momentum=bn_momentum)
        self.spp = nn.Sequential()
        # self.spp.add_module("SPP_BN", _BNReluConv(channels_in, spp_channels,
        #                                           kernel_size=1, bn_momentum=bn_momentum, use_bn=use_bn))
        for i in range(level_num):
            self.spp.add_module('SPP'+str(i), _BNReluConv(spp_channels, level_channels,
                                                          kernel_size=1, bn_momentum=bn_momentum))
        final_cat_channels = spp_channels+level_num*level_channels
        # self.spp.add_module('SPP_FUSE', _BNReluConv(final_cat_channels, channels_out,
        #                                             kernel_size=1, bn_momentum=bn_momentum, use_bn=use_bn))

        self.SPP_FUSE = _BNReluConv(final_cat_channels, channels_out,
                                    kernel_size=1, bn_momentum=bn_momentum)

    def forward(self, x):
        feature_size = x.size()[2:4]
        width_divide_height = feature_size[1]/feature_size[0]
        spp_base = self.SPP_BN(x)

        levels = []
        levels.append(spp_base)

        for i in range(self.level_num):
            pool_dst_size = (self.grid[i], max(1, round(self.grid[i]*width_divide_height)))
            spp_pool = F.adaptive_avg_pool2d(spp_base, pool_dst_size)
            level = self.spp[i].forward(spp_pool)
            level = upsample(level, feature_size)
            levels.append(level)

        spp_cat = torch.cat(levels, 1)
        out = self.SPP_FUSE(spp_cat)
        return out


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        skip = out+identity
        out = self.relu(skip)

        return out, skip


class ResNet(nn.Module):

    def __init__(self, block, layers, feature_channels=128, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.spp_channels = feature_channels
        self.spp_level_num = 3
        self.spp_level_channels = self.spp_channels // self.spp_level_num

        upsample_layers = []
        self.layer1 = self._make_layer(block, 64, layers[0])
        upsample_layers += [_Upsample(self.spp_channels, 64, self.spp_channels, kernel_size=3)]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        upsample_layers += [_Upsample(self.spp_channels, 128, self.spp_channels, kernel_size=3)]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        upsample_layers += [_Upsample(self.spp_channels, 256, self.spp_channels, kernel_size=3)]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.spp = MySpatialPyramidPooling(channels_in=self.inplanes, channels_out=self.spp_channels,
                                           level_num=self.spp_level_num, spp_channels=self.spp_channels,
                                           level_channels=self.spp_level_channels,
                                           grid=(8, 4, 2, 1), bn_momentum=0.01/2)

        self.upsample_layers = nn.ModuleList(list(reversed(upsample_layers)))
        self.out_feature_channels = feature_channels

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.fine_tune = [self.conv1, self.maxpool, self.layer1,
                          self.layer2, self.layer3, self.layer4, self.bn1]
        self.random_init = [self.spp, self.upsample_layers]

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes)
            )

        layers = [block(self.inplanes, planes, stride, downsample, norm_layer)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, skip1 = self.layer1(x)
        x, skip2 = self.layer2(x)
        x, skip3 = self.layer3(x)
        x, skip = self.layer4(x)

        feature = self.spp(skip)
        for skip, up_module in zip([skip3, skip2, skip1], self.upsample_layers):
            feature = up_module(feature, skip)

        return feature


def resnet18(pretrained=True, **kwargs):
        """Constructs a ResNet-18 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        return model


def resnet34(pretrained=True, **kwargs):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
        return


class SwiftNet(nn.Module):

    def __init__(self, backbone, num_classes):
        super(SwiftNet, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.logits = _BNReluConv(self.backbone.out_feature_channels, self.num_classes, batch_norm=True)

    def forward(self, images):
        image_size = list(images.shape)[2:]
        featrues = self.backbone(images)
        logits = self.logits.forward(featrues)
        return upsample(logits, image_size)

    def fine_tune_params(self):
        return list(self.backbone.fine_tune_params())

    def random_init_params(self):
        return list(chain(*([self.logits.parameters(), self.backbone.random_init_params()])))


if __name__ == '__main__':
    resnet = resnet18(pretrained=False)
    model = SwiftNet(resnet, 21)
    for k in model.random_init_params():
        print(type(k.data), k.size())

