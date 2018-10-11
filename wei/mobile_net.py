import torch.nn as nn
import torch.nn.functional as f


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # Depthwise Convolutions.
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # Pointwise Convolutions.
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.base_net = nn.Sequential(
            #                             Layer         Sequence    Output Size
            conv_bn(3, 32, 2),          # Layer 1       Seq 1       150x150x32
            conv_dw(32, 64, 1),         # Layer 2 - 3   Seq 2       150x150x64

            conv_dw(64, 128, 2),        # Layer 4 - 5   Seq 3       75x75x128
            conv_dw(128, 128, 1),       # Layer 6 - 7   Seq 4       75x75x128

            conv_dw(128, 256, 2),       # Layer 8 - 9   Seq 5       38x38x256
            conv_dw(256, 256, 1),       # Layer 10 - 11 Seq 6       38x38x256   <- Regressor & Classifier

            conv_dw(256, 512, 2),       # Layer 12 - 13 Seq 7       19x19x512
            conv_dw(512, 512, 1),       # Layer 14 - 15 Seq 8       19x19x512
            conv_dw(512, 512, 1),       # Layer 16 - 17 Seq 9       19x19x512
            conv_dw(512, 512, 1),       # Layer 18 - 19 Seq 10      19x19x512
            conv_dw(512, 512, 1),       # Layer 20 - 21 Seq 11      19x19x512
            conv_dw(512, 512, 1),       # Layer 22 - 23 Seq 12      19x19x512   <- Regressor & Classifier

            conv_dw(512, 1024, 2),      # Layer 24 - 25 Seq 13      10x10x1024
            conv_dw(1024, 1024, 1)      # Layer 26 - 27 Seq 14      10x10x1024  <- Regressor & Classifier

                                        # Layer 8 (FC Removed)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = f.avg_pool2d(x, 7)
        return x
