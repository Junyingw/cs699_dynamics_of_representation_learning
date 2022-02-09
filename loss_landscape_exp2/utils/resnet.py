"""
ResNet implementation for CIFAR datasets
    Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385

    Adopted from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# SIREN 
class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
       
        return torch.sin(self.w0 * x)


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

    def __init__(self, activation, in_planes, planes, stride, remove_skip_connections):
        super(BasicBlock, self).__init__()
        self.remove_skip_connections = remove_skip_connections

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation = activation

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
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.remove_skip_connections:
            out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, activation="relu", num_classes=10, remove_skip_connections=False):
        super(ResNet, self).__init__()

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = Sine(w0=1.)

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
                    self.activation, self.in_planes, planes, stride,
                    remove_skip_connections=self.remove_skip_connections
                )
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
    
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.squeeze(3).squeeze(2)
        out = self.linear(out)
        
        return out


def resnet20(activation="relu", num_classes=10, remove_skip_connections=False):
    return ResNet(
        BasicBlock, 3, activation=activation, num_classes=num_classes, remove_skip_connections=remove_skip_connections
    )


def resnet32(activation="relu", num_classes=10, remove_skip_connections=False):
    return ResNet(
        BasicBlock, 5, activation=activation, num_classes=num_classes, remove_skip_connections=remove_skip_connections
    )


def resnet44(activation="relu", num_classes=10, remove_skip_connections=False):
    return ResNet(
        BasicBlock, 7, activation=activation, num_classes=num_classes, remove_skip_connections=remove_skip_connections
    )


def resnet56(activation="relu", num_classes=10, remove_skip_connections=False):
    return ResNet(
        BasicBlock, 9, activation=activation, num_classes=num_classes, remove_skip_connections=remove_skip_connections
    )


def get_resnet(model_string, activation, remove_skip_connections):
    if model_string == "resnet56":
        return resnet56(activation=activation, remove_skip_connections=remove_skip_connections)

    if model_string == "resnet44":
        return resnet44(activation=activation, remove_skip_connections=remove_skip_connections)

    if model_string == "resnet32":
        return resnet32(activation=activation, remove_skip_connections=remove_skip_connections)

    if model_string == "resnet20":
        return resnet20(activation=activation, remove_skip_connections=remove_skip_connections)


if __name__ == "__main__":
    import torch

    image = torch.rand(10, 3, 32, 32)
    model = resnet20(activation="siren")
    output = model(image)
    print(output.shape)
