import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        in_channels = config.get("stack_length", 1)
        self.conv1 = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 9, kernel_size=3, padding=1),
            nn.BatchNorm2d(9),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(9, 18, kernel_size=3, padding=1),
            nn.BatchNorm2d(18),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(18, 22, kernel_size=3, padding=1),
            nn.BatchNorm2d(22),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(22, 26, kernel_size=3, padding=1),
            nn.BatchNorm2d(26),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(2),
            # Block 5
            nn.Conv2d(26, 30, kernel_size=3, padding=1),
            nn.BatchNorm2d(30),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(2),
            # Block 6 (final)
            nn.Conv2d(30, 34, kernel_size=3, padding=1),
            nn.BatchNorm2d(34),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.4),
        )

        with torch.no_grad():
            if config.get("grey_scale", True):
                x = torch.ones(1, 1, config["size_x"], config["size_y"])
            else:
                x = torch.ones(1, 3, config["size_x"], config["size_y"])

            out = self.conv1(x)
            features = out.numel()

        self.lin1 = nn.Sequential(
            nn.Linear(features, 500),
            nn.Linear(500, 200),
            nn.Linear(200, 2),
        )

    def forward(self, img):
        output = self.conv1(img)
        output = output.view(output.size(0), -1)

        output = self.lin1(output)
        return output


# TODO: Add a convolution and de-convolution network architecture
class ConvDeconvNet(nn.Module):
    def __init__(self, config):
        super(ConvDeconvNet, self).__init__()
        self.config = config
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                config["stack_length"], 32, kernel_size=8, stride=4, padding=4
            ),  # 80x160 -> 21x41
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # 21x41 -> 10x20
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 10x20 -> 5x10
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # 5x10 -> 2x5
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 2x5 -> 2x5
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        # Decoder (pure transposed convolutions)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 2), stride=(1, 2)),  # 3x2 -> 5x4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=(3, 4), stride=(3, 2)
            ),  # 5x4 -> 15x10
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=(7, 2), stride=(2, 2)
            ),  # 15x10 -> 35x20
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                16, 1, kernel_size=(3, 4), stride=(3, 4)
            ),  # 35x20 -> 105x80
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
