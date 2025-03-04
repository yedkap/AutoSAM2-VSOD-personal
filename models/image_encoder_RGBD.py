import warnings
import os
import pandas as pd
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = ['ResNet', 'ResNet18', 'ResNet34', 'ResNet50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None,
                 activation=nn.ReLU(inplace=True), residual_only=False):
        super(BasicBlock, self).__init__()
        self.residual_only = residual_only
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and '
                             'base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.act = activation
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.residual_only:
            return out
        out = out + identity
        out = self.act(out)

        return out


class NonBottleneck1D(nn.Module):
    """
    ERFNet-Block
    Paper:
    http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf
    Implementation from:
    https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=None, dilation=1, norm_layer=None,
                 activation=nn.ReLU(inplace=True), residual_only=False):
        super().__init__()
        warnings.warn('parameters groups, base_width and norm_layer are '
                      'ignored in NonBottleneck1D')
        dropprob = 0
        self.conv3x1_1 = nn.Conv2d(inplanes, planes, (3, 1),
                                   stride=(stride, 1), padding=(1, 0),
                                   bias=True)
        self.conv1x3_1 = nn.Conv2d(planes, planes, (1, 3),
                                   stride=(1, stride), padding=(0, 1),
                                   bias=True)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-03)
        self.act = activation
        self.conv3x1_2 = nn.Conv2d(planes, planes, (3, 1),
                                   padding=(1 * dilation, 0), bias=True,
                                   dilation=(dilation, 1))
        self.conv1x3_2 = nn.Conv2d(planes, planes, (1, 3),
                                   padding=(0, 1 * dilation), bias=True,
                                   dilation=(1, dilation))
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)
        self.downsample = downsample
        self.stride = stride
        self.residual_only = residual_only

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.act(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.act(output)

        output = self.conv3x1_2(output)
        output = self.act(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        if self.downsample is None:
            identity = input
        else:
            identity = self.downsample(input)

        if self.residual_only:
            return output
        # +input = identity (residual connection)
        return self.act(output + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None,
                 activation=nn.ReLU(inplace=True)):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # both self.conv2 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.act = activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.act(out)
        return out


class ResNet(nn.Module):

    def __init__(self, layers, block,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, dilation=None,
                 norm_layer=None, input_channels=3,
                 activation=nn.ReLU(inplace=True)):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        self.replace_stride_with_dilation = replace_stride_with_dilation
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got "
                             "{}".format(replace_stride_with_dilation))
        if dilation is not None:
            dilation = dilation
            if len(dilation) != 4:
                raise ValueError("dilation should be None "
                                 "or a 4-element tuple, got "
                                 "{}".format(dilation))
        else:
            dilation = [1, 1, 1, 1]

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_channels, self.inplanes,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.act = activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.down_2_channels_out = 64
        if self.replace_stride_with_dilation == [False, False, False]:
            self.down_4_channels_out = 64 * block.expansion
            self.down_8_channels_out = 128 * block.expansion
            self.down_16_channels_out = 256 * block.expansion
            self.down_32_channels_out = 512 * block.expansion
        elif self.replace_stride_with_dilation == [False, True, True]:
            self.down_4_channels_out = 64 * block.expansion
            self.down_8_channels_out = 512 * block.expansion

        self.layer1 = self._make_layer(
            block, 64, layers[0], dilate=dilation[0]
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1],
            stride=2, dilate=dilation[1],
            replace_stride_with_dilation=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2],
            stride=2, dilate=dilation[2],
            replace_stride_with_dilation=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3],
            stride=2, dilate=dilation[3],
            replace_stride_with_dilation=replace_stride_with_dilation[2]
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual
        # block behaves like an identity. This improves the model by 0.2~0.3%
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks,
                    stride=1, dilate=1, replace_stride_with_dilation=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if replace_stride_with_dilation:
            self.dilation *= stride
            stride = 1
        if dilate > 1:
            self.dilation = dilate
            dilate_first_block = dilate
        else:
            dilate_first_block = previous_dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, dilate_first_block,
                            norm_layer,
                            activation=self.act))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer,
                                activation=self.act))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x_down2 = self.act(x)
        x = self.maxpool(x_down2)

        x_layer1 = self.forward_resblock(x, self.layer1)
        x_layer2 = self.forward_resblock(x_layer1, self.layer2)
        x_layer3 = self.forward_resblock(x_layer2, self.layer3)
        x_layer4 = self.forward_resblock(x_layer3, self.layer4)

        if self.replace_stride_with_dilation == [False, False, False]:
            features = [x_layer4, x_layer3, x_layer2, x_layer1]

            self.skip3_channels = x_layer3.size()[1]
            self.skip2_channels = x_layer2.size()[1]
            self.skip1_channels = x_layer1.size()[1]
        elif self.replace_stride_with_dilation == [False, True, True]:
            # x has resolution 1/8
            # skip4 has resolution 1/8
            # skip3 has resolution 1/8
            # skip2 has resolution 1/8
            # skip1 has resolution 1/4
            # x_down2 has resolution 1/2
            features = [x, x_layer1, x_down2]

            self.skip3_channels = x_layer3.size()[1]
            self.skip2_channels = x_layer2.size()[1]
            self.skip1_channels = x_layer1.size()[1]

        return features

    def forward_resblock(self, x, layers):
        for l in layers:
            x = l(x)
        return x

    def forward_first_conv(self, x):
        # be aware that maxpool still needs to be applied after this function
        # and before forward_layer1()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        return x

    def forward_layer1(self, x):
        # be ware that maxpool still needs to be applied after
        # forward_first_conv() and before this function
        x = self.forward_resblock(x, self.layer1)
        self.skip1_channels = x.size()[1]
        return x

    def forward_layer2(self, x):
        x = self.forward_resblock(x, self.layer2)
        self.skip2_channels = x.size()[1]
        return x

    def forward_layer3(self, x):
        x = self.forward_resblock(x, self.layer3)
        self.skip3_channels = x.size()[1]
        return x

    def forward_layer4(self, x):
        x = self.forward_resblock(x, self.layer4)
        return x


def ResNet18(pretrained=False,
             pretrained_dir='./trained_models/imagenet',
             **kwargs):
    if 'block' not in kwargs:
        kwargs['block'] = BasicBlock
    else:
        # convert string to block object
        kwargs['block'] = eval(kwargs['block'])
    model = ResNet([2, 2, 2, 2], **kwargs)
    if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
        input_channels = 1
    else:
        input_channels = 3
    if kwargs['block'] != BasicBlock and pretrained:
        model = load_pretrained_with_different_encoder_block(
            model, kwargs['block'].__name__,
            input_channels, 'r18',
            pretrained_dir=pretrained_dir
        )
    elif pretrained:
        weights = model_zoo.load_url(model_urls['resnet18'], model_dir='../segment_anything_1/modeling/')
        if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
            # sum the weights of the first convolution
            weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
                                                axis=1, keepdim=True)
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        model.load_state_dict(weights, strict=True)
        print('Loaded ResNet18 pretrained on ImageNet')
    return model


def ResNet34(pretrained=False,
             pretrained_dir='./trained_models/imagenet',
             **kwargs):
    if 'block' not in kwargs:
        kwargs['block'] = BasicBlock
    else:
        if kwargs['block'] in globals():
            # convert string to block object
            kwargs['block'] = globals()[kwargs['block']]
        else:
            raise NotImplementedError('Block {} is not implemented'
                                      ''.format(kwargs['block']))
    if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
        input_channels = 1
    else:
        input_channels = 3
    model = ResNet([3, 4, 6, 3], **kwargs)
    if kwargs['block'] != BasicBlock and pretrained:
        model = load_pretrained_with_different_encoder_block(
            model, kwargs['block'].__name__,
            input_channels, 'r34',
            pretrained_dir=pretrained_dir
        )
    elif pretrained:
        weights = model_zoo.load_url(model_urls['resnet34'], model_dir='../segment_anything_1/modeling/')
        if input_channels == 1:
            # sum the weights of the first convolution
            weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
                                                axis=1, keepdim=True)
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        model.load_state_dict(weights, strict=True)
        print('Loaded ResNet34 pretrained on ImageNet')
    return model


def ResNet50(pretrained=False, **kwargs):
    model = ResNet([3, 4, 6, 3], Bottleneck, **kwargs)
    if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
        input_channels = 1
    else:
        input_channels = 3
    if pretrained:
        weights = model_zoo.load_url(model_urls['resnet50'], model_dir='../segment_anything_1/modeling/')
        if input_channels == 1:
            # sum the weights of the first convolution
            weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
                                                axis=1, keepdim=True)
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        model.load_state_dict(weights, strict=True)
        print('Loaded ResNet50 pretrained on ImageNet')
    return model


def load_pretrained_with_different_encoder_block(
        model, encoder_block, input_channels, resnet_name,
        pretrained_dir='./trained_models/imagenet'):

    ckpt_path = os.path.join(pretrained_dir, f'{resnet_name}_NBt1D.pth')

    if not os.path.exists(ckpt_path):
        # get best weights file from logs
        logs = pd.read_csv(os.path.join(pretrained_dir, 'logs.csv'))
        idx_top1 = logs['acc_val_top-1'].idxmax()
        acc_top1 = logs['acc_val_top-1'][idx_top1]
        epoch = logs.epoch[idx_top1]
        ckpt_path = os.path.join(pretrained_dir,
                                 'ckpt_epoch_{}.pth'.format(epoch))
        print(f"Choosing checkpoint {ckpt_path} with top1 acc {acc_top1}")

    # load weights
    if torch.cuda.is_available():
        checkpoint = torch.load(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint['state_dict2'] = OrderedDict()

    # rename keys and leave out last fully connected layer
    for key in checkpoint['state_dict']:
        if 'encoder' in key:
            checkpoint['state_dict2'][key.split('encoder.')[-1]] = \
                checkpoint['state_dict'][key]
    weights = checkpoint['state_dict2']

    if input_channels == 1:
        # sum the weights of the first convolution
        weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
                                            axis=1,
                                            keepdim=True)

    model.load_state_dict(weights, strict=False)
    print(f'Loaded {resnet_name} with encoder block {encoder_block} '
          f'pretrained on ImageNet')
    print(ckpt_path)
    return model
class ConvBNAct(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size,
                 activation=nn.ReLU(inplace=True), dilation=1, stride=1):
        super(ConvBNAct, self).__init__()
        padding = kernel_size // 2 + dilation - 1
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=False,
                                          dilation=dilation,
                                          stride=stride))
        self.add_module('bn', nn.BatchNorm2d(channels_out))
        self.add_module('act', activation)



class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


def swish(x):
    return x * torch.sigmoid(x)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class ImageEncoderRGB_D(nn.Module):
    def __init__(self,encoder_rgb = 'resnet34',
        height=480,
        width=640,
        num_classes=37,
        encoder_depth = 'resnet34',
        encoder_block = 'BasicBlock',
        channels_decoder = None,  # default: [128, 128, 128]
        pretrained = False,
        pretrained_dir = 'cp_RGBD',
        activation = 'relu',
        nr_decoder_blocks = None,  # default: [1, 1, 1]
        fuse_depth_in_rgb_encoder = 'SE-add',
        upsampling = 'bilinear'):
            super(ImageEncoderRGB_D, self).__init__()

            self.fuse_depth_in_rgb_encoder = fuse_depth_in_rgb_encoder

            # set activation function
            if activation.lower() == 'relu':
                self.activation = nn.ReLU(inplace=True)
            elif activation.lower() in ['swish', 'silu']:
                self.activation = Swish()
            elif activation.lower() == 'hswish':
                self.activation = Hswish()
            else:
                raise NotImplementedError(
                    'Only relu, swish and hswish as activation function are '
                    'supported so far. Got {}'.format(activation))

            if encoder_rgb == 'resnet50' or encoder_depth == 'resnet50':
                warnings.warn('Parameter encoder_block is ignored for ResNet50. '
                              'ResNet50 always uses Bottleneck')
            channels_decoder = [256, 128, 64]
            #channels_decoder = [64,128,256]
            nr_decoder_blocks = [1, 1, 1]
            # rgb encoder
            if encoder_rgb == 'resnet18':
                self.encoder_rgb = ResNet18(
                    block=encoder_block,
                    pretrained=pretrained,
                    pretrained_dir=pretrained_dir,
                    activation=self.activation)
            elif encoder_rgb == 'resnet34':
                self.encoder_rgb = ResNet34(
                    block=encoder_block,
                    pretrained=pretrained,
                    pretrained_dir=pretrained_dir,
                    activation=self.activation)
            elif encoder_rgb == 'resnet50':
                self.encoder_rgb = ResNet50(
                    pretrained=pretrained,
                    activation=self.activation)
            else:
                raise NotImplementedError(
                    'Only ResNets are supported for '
                    'encoder_rgb. Got {}'.format(encoder_rgb))

            # depth encoder
            if encoder_depth == 'resnet18':
                self.encoder_depth = ResNet18(
                    block=encoder_block,
                    pretrained=pretrained,
                    pretrained_dir=pretrained_dir,
                    activation=self.activation,
                    input_channels=1)
            elif encoder_depth == 'resnet34':
                self.encoder_depth = ResNet34(
                    block=encoder_block,
                    pretrained=pretrained,
                    pretrained_dir=pretrained_dir,
                    activation=self.activation,
                    input_channels=1)
            elif encoder_depth == 'resnet50':
                self.encoder_depth = ResNet50(
                    pretrained=pretrained,
                    activation=self.activation,
                    input_channels=1)
            else:
                raise NotImplementedError(
                    'Only ResNets are supported for '
                    'encoder_depth. Got {}'.format(encoder_rgb))
            d=[128,128,128]
            self.channels_decoder_in = self.encoder_rgb.down_32_channels_out
            self.skip_layer0 = list()
            self.skip_layer1 = list()
            self.skip_layer2 = list()
            self.skip_layer3 = list()
            if fuse_depth_in_rgb_encoder == 'SE-add':
                self.se_layer0 = SqueezeAndExciteFusionAdd(
                    64, activation=self.activation)
                self.se_layer1 = SqueezeAndExciteFusionAdd(
                    self.encoder_rgb.down_4_channels_out,
                    activation=self.activation)
                self.se_layer2 = SqueezeAndExciteFusionAdd(
                    self.encoder_rgb.down_8_channels_out,
                    activation=self.activation)
                self.se_layer3 = SqueezeAndExciteFusionAdd(
                    self.encoder_rgb.down_16_channels_out,
                    activation=self.activation)
                self.se_layer4 = SqueezeAndExciteFusionAdd(
                    self.encoder_rgb.down_32_channels_out,
                    activation=self.activation)
            layers_skip1 = list()
            d[0]=self.encoder_rgb.down_4_channels_out
            if self.encoder_rgb.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    self.encoder_rgb.down_4_channels_out,
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            d[1] = self.encoder_rgb.down_8_channels_out
            if self.encoder_rgb.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    self.encoder_rgb.down_8_channels_out,
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            d[2] = self.encoder_rgb.down_16_channels_out
            if self.encoder_rgb.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    self.encoder_rgb.down_16_channels_out,
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)


    def forward(self, rgb, depth):

        rgb = self.encoder_rgb.forward_first_conv(rgb)
        depth = self.encoder_depth.forward_first_conv(depth)

        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer0(rgb, depth)

        rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)
        depth = self.encoder_depth.forward_layer1(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer1(rgb, depth)
        skip1 = self.skip_layer1(fuse)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse)
        depth = self.encoder_depth.forward_layer2(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer2(rgb, depth)
        skip2 = self.skip_layer2(fuse)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse)
        depth = self.encoder_depth.forward_layer3(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer3(rgb, depth)
        skip3 = self.skip_layer3(fuse)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse)
        depth = self.encoder_depth.forward_layer4(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer4(rgb, depth)

        #out = self.context_module(fuse)
        #out = self.decoder(enc_outs=[out, skip3, skip2, skip1])
        print(rgb.shape)
        print(depth.shape)
        out=fuse

        skip1_upscaled=F.interpolate(skip1, size=(128, 128), mode='bilinear', align_corners=False)
        skip2_upscaled = F.interpolate(skip2, size=(128, 128), mode='bilinear', align_corners=False)
        skip3_upscaled = F.interpolate(skip3, size=(128, 128), mode='bilinear', align_corners=False)
        out_upscaled= F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)


        #return  out_upscaled,skip1_upscaled,skip2_upscaled,skip3_upscaled
        return out, skip1, skip2, skip3



class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SqueezeAndExcitationTensorRT(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitationTensorRT, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # TensorRT restricts the maximum kernel size for pooling operations
        # by "MAX_KERNEL_DIMS_PRODUCT" which leads to problems if the input
        # feature maps are of large spatial size
        # -> workaround: use cascaded two-staged pooling
        # see: https://github.com/onnx/onnx-tensorrt/issues/333
        if x.shape[2] > 120 and x.shape[3] > 160:
            weighting = F.adaptive_avg_pool2d(x, 4)
        else:
            weighting = x
        weighting = F.adaptive_avg_pool2d(weighting, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out