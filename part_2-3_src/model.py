import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, network=1, in_channels=1, out_channels=32, kernel_size=3, drop_out=0.2) -> None:
        super(ConvNet, self).__init__()
        # Variable for the Conv2d. If BatchNorm2d is used then no need to use bias
        bias = False
        # Variable for the network choice
        self.networkChoice = network

        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=out_channels, out_channels=(out_channels * 2), kernel_size=kernel_size, stride=1,
                      padding=1, bias=bias),
            nn.BatchNorm2d((out_channels * 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=(out_channels * 2), out_channels=(out_channels * 4), kernel_size=kernel_size,
                      stride=1, padding=1, bias=bias),
            nn.BatchNorm2d((out_channels * 4)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear((out_channels * 4) * (self.__calculateInputForLinearLayers(224, self.conv_layers1)) ** 2,
                      (out_channels * 4)),
            nn.Dropout(drop_out),
            nn.Linear((out_channels * 4), 4)
        )

        self.network1 = nn.Sequential(
            self.conv_layers1,
            self.linear_layers1
        )

        self.conv_layers2 = nn.Sequential(

            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=(out_channels), out_channels=(out_channels * 2), kernel_size=kernel_size, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=(out_channels * 2), out_channels=(out_channels * 4), kernel_size=kernel_size,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=(out_channels * 4), out_channels=(out_channels * 4), kernel_size=kernel_size,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=(out_channels * 4), out_channels=(out_channels * 8), kernel_size=kernel_size,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=(out_channels * 8), out_channels=(out_channels * 8), kernel_size=kernel_size,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.linear_layers2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear((out_channels * 8) * (self.__calculateInputForLinearLayers(224, self.conv_layers2)) ** 2,
                      (out_channels * 32)),
            nn.ReLU(),
            nn.Linear((out_channels * 32), (out_channels * 16)),
            nn.ReLU(),
            nn.Linear((out_channels * 16), 4)
        )

        self.network2 = nn.Sequential(
            self.conv_layers2,
            self.linear_layers2
        )

        self.conv_layers3 = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, groups=in_channels, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Pointwise convolution
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Additional depthwise separable convolutions
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=kernel_size, padding=1, groups=out_channels * 2,
                      bias=bias),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.linear_layers3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear((out_channels * 4) * (self.__calculateInputForLinearLayers(224, self.conv_layers3)) ** 2,
                      out_channels * 16),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(out_channels * 16, out_channels * 8),
            nn.ReLU(),
            nn.Linear(out_channels * 8, 4)  # Final output layer
        )
        self.network3 = nn.Sequential(
            self.conv_layers3,
            self.linear_layers3
        )

    # Calculates the input size that will be fed into the linear layers based on the structure of the convolutional
    # and max pooling layers
    def __calculateInputForLinearLayers(self, input, layers) -> int:

        for idx in range(len(layers)):
            layer = layers[idx]
            kernel_size = None
            stride = None
            padding = None
            if isinstance(layer, nn.Conv2d):
                if hasattr(layer, 'kernel_size'):
                    kernel_size = layer.kernel_size[0]
                if hasattr(layer, 'stride'):
                    stride = layer.stride[0]
                if hasattr(layer, 'padding'):
                    padding = layer.padding[0]
                input = (input - kernel_size + 2 * padding) / (stride) + 1
            if isinstance(layer, nn.MaxPool2d):
                if hasattr(layer, 'kernel_size'):
                    kernel_size = layer.kernel_size
                if hasattr(layer, 'stride'):
                    stride = layer.stride
                if hasattr(layer, 'padding'):
                    padding = layer.padding
                input = (input - kernel_size) / stride + 1
        return int(input)

    def forward(self, x):
        if self.networkChoice == 1:
            x = self.network1(x)
        elif self.networkChoice == 2:
            x = self.network2(x)
        else:
            x = self.network3(x)
        return x

    # Describes the network used
    def describeNetwork(self):
        if self.networkChoice == 1:
            print(self.network1)
        elif self.networkChoice == 2:
            print(self.network2)
        else:
            print(self.network3)
