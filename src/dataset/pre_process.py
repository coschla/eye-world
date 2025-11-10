from collections import deque

import torch
from torchvision import transforms


class Resize:
    def __init__(self, config):
        if config.get("grey_scale", True):
            self.transform = transforms.Compose(
                [
                    transforms.Resize((config["size_x"], config["size_y"])),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((config["size_x"], config["size_y"])),
                    transforms.ToTensor(),
                ]
            )

    def __call__(self, sample):
        img, eye_gazes = sample
        img = self.transform(img)

        return img, torch.tensor(eye_gazes[-1], dtype=torch.float)


class Stack:
    def __init__(self, config):
        # TODO: Add the logic to stack the images here
        self.stack_len = config["stack_length"]
        self.stack = deque(maxlen=self.stack_len)

    def __call__(self, sample):
        img, eye_gazes = sample
        # TODO: Return a stack of images.
        # Add a parameter of the config where we can set the stack length

        if len(self.stack) < self.stack_len:
            while len(self.stack) < self.stack_len:
                self.stack.append(img)
        else:
            self.stack.append(img)
        stacked = torch.cat(list(self.stack), dim=0)

        return stacked, eye_gazes


class ComposePreprocessor:
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def __call__(self, sample):
        for p in self.preprocessors:
            sample = p(sample)
        return sample
