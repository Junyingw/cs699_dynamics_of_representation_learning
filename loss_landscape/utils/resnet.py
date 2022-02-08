"""
ResNet implementation for CIFAR datasets
    Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385

    Adopted from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
"""

import torch.nn as nn
import torch.nn.functional as F


class DownSample2D(nn.Module):
    def __init__(self, stride_h, stride_w):
        super().__init__()
        self.stride_h = stride_h
        self.stride_w = stride_w

    def forward(self, x):
        # we expect input to be in NCHW format
        return x[:, :, ::self.stride_h, ::self.stride_w]


class PadChannel2D(nn.Module):
    def __init__(self, pad_size):
        super().__init__()
        self.pad_size = pad_size

    def forward(self, x):
        # we expect input to be in NCHW format
        return F.pad(x, (0, 0, 0, 0, self.pad_size, self.pad_size), "constant", 0.0)


class BasicBlock(nn.Module):
    # expansion = 1

    def __init__(self, in_planes, planes, stride, remove_skip_connections):
        super(BasicBlock, self).__init__()
        self.remove_skip_connections = remove_skip_connections

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if not self.remove_skip_connections:
            self.shortcut = nn.Sequential()

            # Option A from the paper
            if stride != 1 or in_planes != planes:
                assert planes > in_planes and (
                        planes - in_planes) % 2 == 0, "out planes should be more than inplanes"
                # subsample and pad x
                self.shortcut = nn.Sequential(
                    DownSample2D(stride, stride),
                    PadChannel2D((planes - in_planes) // 2)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.remove_skip_connections:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, remove_skip_connections=False):
        super(ResNet, self).__init__()

        self.remove_skip_connections = remove_skip_connections

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.in_planes = 16
        self.layer1 = self._make_layer(block, 16, num_blocks, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks, stride=2)

        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes, planes, stride,
                    remove_skip_connections=self.remove_skip_connections
                )
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.squeeze(3).squeeze(2)
        out = self.linear(out)
        return out


def resnet20(num_classes=10, remove_skip_connections=False):
    return ResNet(
        BasicBlock, 3, num_classes=num_classes, remove_skip_connections=remove_skip_connections
    )


def resnet32(num_classes=10, remove_skip_connections=False):
    return ResNet(
        BasicBlock, 5, num_classes=num_classes, remove_skip_connections=remove_skip_connections
    )


def resnet44(num_classes=10, remove_skip_connections=False):
    return ResNet(
        BasicBlock, 7, num_classes=num_classes, remove_skip_connections=remove_skip_connections
    )


def resnet56(num_classes=10, remove_skip_connections=False):
    return ResNet(
        BasicBlock, 9, num_classes=num_classes, remove_skip_connections=remove_skip_connections
    )

"""
Fixed-ResNet implementation for CIFAR datasets
    Reference:
    [1] Hongyi Zhang, Yann N. Dauphin, Tengyu Ma. 
        Fixup Initialization: Residual Learning Without Normalization.
        7th International Conference on Learning Representations (ICLR 2019).
    Adopted from https://github.com/hongyi-zhang/Fixup/blob/master/cifar/models/fixup_resnet_cifar.py
"""

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class FixupResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(FixupResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x + self.bias1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x + self.bias2)

        return x


def fixup_resnet20(num_classes=10, remove_skip_connections=False):
    """Constructs a Fixup-ResNet-20 model.
    """
    model = FixupResNet(FixupBasicBlock, [3, 3, 3])
    return model


def fixup_resnet32(num_classes=10, remove_skip_connections=False):
    """Constructs a Fixup-ResNet-32 model.
    """
    model = FixupResNet(FixupBasicBlock, [5, 5, 5])
    return model


def fixup_resnet44(num_classes=10, remove_skip_connections=False):
    """Constructs a Fixup-ResNet-44 model.
    """
    model = FixupResNet(FixupBasicBlock, [7, 7, 7])
    return model


def fixup_resnet56(num_classes=10, remove_skip_connections=False):
    """Constructs a Fixup-ResNet-56 model.
    """
    model = FixupResNet(FixupBasicBlock, [9, 9, 9])
    return model

def get_resnet(model_string):
    if model_string == "resnet56":
        return resnet56

    if model_string == "resnet44":
        return resnet44

    if model_string == "resnet32":
        return resnet32

    if model_string == "resnet20":
        return resnet20

    if model_string == "fixup_resnet56":
        return fixup_resnet56

    if model_string == "fixup_resnet44":
        return fixup_resnet44

    if model_string == "fixup_resnet32":
        return fixup_resnet32

    if model_string == "fixup_resnet20":
        return fixup_resnet20

if __name__ == "__main__":
    import torch

    image = torch.rand(10, 3, 32, 32)
    model = resnet20()
    output = model(image)
    print(output.shape)
