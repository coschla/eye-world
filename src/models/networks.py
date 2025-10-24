import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=3, padding=1),
            nn.BatchNorm2d(9),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(9, 27, kernel_size=3, padding=1),
            nn.BatchNorm2d(27),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(27, 9, kernel_size=3, padding=1),
            nn.BatchNorm2d(9),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(9, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
        )
        with torch.no_grad():
            x = torch.ones(1, 3, config["size_x"], config["size_y"])
            out = self.conv1(x)
            features = out.numel()
        self.lin1 = nn.Linear(features, 500)
        self.lin2 = nn.Linear(500, 100)
        self.lin3 = nn.Linear(100, 2)

        self.relu = nn.ReLU()

    # TODO: Implement the network architecture here

    def forward(self, img):
        output = self.conv1(img)
        output = output.view(output.size(0), -1)
        output = self.relu(self.lin1(output))
        output = self.relu(self.lin2(output))
        output = self.lin3(output)
        return output
        # TODO: Impelement the forward propogation
