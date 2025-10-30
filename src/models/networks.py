import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 9, kernel_size=3, padding=1),
            nn.BatchNorm2d(9),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(9, 18, kernel_size=3, padding=1),
            nn.BatchNorm2d(18),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(18, 27, kernel_size=3, padding=1),
            nn.BatchNorm2d(27),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(27, 27, kernel_size=3, padding=1),
            nn.BatchNorm2d(27),
            nn.LeakyReLU(),
            # nn.Dropout2d(p=0.3),
            # nn.MaxPool2d(2),
            # nn.Conv2d(18, 9, kernel_size=3, padding=1),
            # nn.BatchNorm2d(9),
            # nn.LeakyReLU(),
            # nn.Dropout2d(p=0.3),
            # nn.MaxPool2d(2),
            # nn.Conv2d(9, 1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(1),
            # nn.LeakyReLU(),
        )

        with torch.no_grad():
            if config.get("grey_scale", True):
                x = torch.ones(1, 1, config["size_x"], config["size_y"])
            else:
                x = torch.ones(1, 3, config["size_x"], config["size_y"])
            print(x)
            out = self.conv1(x)
            features = out.numel()
<<<<<<< HEAD

=======
        self.linear = nn.Sequential(
            nn.Linear(features, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 25),
            nn.LeakyReLU(),
            nn.Linear(25, 2),
        )
>>>>>>> dd199f7 (frame visulization created)
        self.lin1 = nn.Linear(features, 500)

    def forward(self, img):
        output = self.conv1(img)
        output = output.view(output.size(0), -1)

        output = self.linear(output)
        return output
