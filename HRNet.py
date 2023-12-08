# -*- coding: utf-8 -*-

"""
Model definition
"""

__status__ = "Dev"


import torch
from torchvision.models import densenet121
import torch.nn as nn
from ConvLSTM_pytorch.convlstm import ConvLSTM


def scale_as(x, y):
    '''
    scale x to the same size as y
    '''
    y_size = y.size(2), y.size(3)

    x_scaled = torch.nn.functional.interpolate(
        x, size=y_size, mode='bilinear',
        align_corners=True)
    return x_scaled


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # print("Initializing Conv Weights...")
        nn.init.xavier_normal_(m.weight.data)
    if isinstance(m, nn.ConvTranspose2d):
        # print("Initializing ConvTranspose weights...")
        nn.init.xavier_normal_(m.weight.data)
    if isinstance(m, nn.Linear):
        # print("Initializing Linear Weights...")
        nn.init.xavier_normal_(m.weight.data)
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    # print("Finished Initializing.")


class Trunk(nn.Module):
    def __init__(self, pretrained=False, in_channels=1):
        super().__init__()
        self.pretrained = pretrained

        self.backbone = densenet121(pretrained=pretrained).features
        self.backbone.conv0 = nn.Conv2d(in_channels, 64,
                                        kernel_size=(7, 7), stride=(2, 2),
                                        padding=(3, 3), bias=False)

        # Initialize the newly created layer using Xavier Normal
        self.backbone.conv0.apply(weights_init)

        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-5]))

    def forward(self, x):
        return self.backbone(x)


class AttnHead(nn.Module):
    def __init__(self, input_channels,middle_channels):
        super().__init__()
        self.input_channels = input_channels
        self.middle_channels = middle_channels

        self.attn = nn.Sequential(
            nn.Conv2d(self.input_channels, self.input_channels,
                      kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(self.input_channels),
            nn.LeakyReLU(negative_slope=0.2,
                         inplace=True),
            nn.Dropout(0.0),

            nn.Conv2d(self.input_channels, self.input_channels,
                      kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(self.input_channels),
            nn.LeakyReLU(negative_slope=0.2,
                         inplace=True),
            nn.Dropout(0.0),

            nn.Conv2d(self.input_channels, self.middle_channels,
                      kernel_size=(1, 1),
                      stride=(1, 1), padding=0, bias=False),

            nn.Sigmoid()
        )

        self.attn.apply(weights_init)

    def forward(self, x):
        return self.attn(x)


class SegHead(nn.Module):
    def __init__(self, input_channels, middle_channels):
        super().__init__()
        self.input_channels = input_channels
        self.middle_channels = middle_channels

        self.seg = nn.Sequential(
                                    nn.Conv2d(self.input_channels, self.input_channels,
                                            kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1), bias=False),
                                    nn.BatchNorm2d(self.input_channels),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                inplace=True),
                                    nn.Dropout(0.0),

                                    nn.Conv2d(self.input_channels, self.input_channels,
                                            kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1), bias=False),
                                    nn.BatchNorm2d(self.input_channels),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                inplace=True),
                                    nn.Dropout(0.0),

                                    nn.Conv2d(self.input_channels, self.middle_channels,
                                            kernel_size=(1, 1), stride=(1, 1),
                                            padding=0, bias=False)
                                )

        self.seg.apply(weights_init)

    def forward(self, x):
        return self.seg(x)


class _DecoderBlock(nn.Module):
    '''
    Decoder Block for Attention Module
    '''
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
                                    nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(middle_channels),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                inplace=True),
                                    nn.Conv2d(middle_channels, middle_channels,
                                            kernel_size=3, padding=1),
                                    nn.BatchNorm2d(middle_channels),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                inplace=True),
                                    nn.ConvTranspose2d(middle_channels, middle_channels,
                                                    kernel_size=2, stride=2),
                                    nn.BatchNorm2d(middle_channels),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                inplace=True),
                                    nn.ConvTranspose2d(middle_channels, out_channels,
                                                    kernel_size=2, stride=2)
                                )

        self.decode.apply(weights_init)

    def forward(self, x):
        return self.decode(x)


class HRNet_LSTM(nn.Module):
    '''
    Calcium segmentation model architecture
    '''
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.convlstm = ConvLSTM(1, 3, (3, 3), 2, True, True, False)
        self.convlstm.apply(weights_init)
        self.Attn = AttnHead(input_channels=3,middle_channels=self.num_classes)
        self.sub_seg = SegHead(
                                input_channels=3,
                                middle_channels=self.num_classes
                                )
        self.main_trunk = Trunk(
                                pretrained=self.pretrained,
                                in_channels=1
                                )
        self.main_seg = SegHead(
                                input_channels=512,
                                middle_channels=self.num_classes
                                )

    def forward(self, x):
        _, _t = self.convlstm(x.unsqueeze_(2))
        a = self.Attn(_t[0][0])
        _s = self.sub_seg(_t[0][0])

        t = self.main_trunk(x[:, 1])
        s = self.main_seg(t)
        
        s = scale_as(s, a)
        return torch.max(_s*(1-a), s*a)


class Heart_HRNet_LSTM(nn.Module):
    '''
    Whole heart segmentaton model architecture
    '''
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.convlstm = ConvLSTM(1, 3 ,(3, 3), 2, True, True, False)
        self.convlstm.apply(weights_init)
        self.Attn = AttnHead(input_channels=3, middle_channels=self.num_classes)
        self.sub_seg = SegHead(input_channels=3,
                               middle_channels=self.num_classes)
        self.main_trunk = Trunk(pretrained=self.pretrained, in_channels=1)
        self.main_seg = SegHead(input_channels=512,
                                middle_channels=self.num_classes)

    def forward(self, x):
        _, _t = self.convlstm(x.unsqueeze_(2))
        a = self.Attn(_t[0][0])
        _s = self.sub_seg(_t[0][0])

        t = self.main_trunk(x[:, 1])
        s = self.main_seg(t)

        s = scale_as(s, a)
        return torch.add(_s * (1 - a), s * a)


if __name__ == "__main__":
    heart_model = Heart_HRNet_LSTM(num_classes=2, pretrained=False)
    lesion_model = HRNet_LSTM(num_classes=3, pretrained=False)
