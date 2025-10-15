import torch
from torchvision import transforms


class ResizePreprocessor:
    def __init__(self, config):
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


class ComposePreprocessor:
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def __call__(self, sample):
        for p in self.preprocessors:
            sample = p(sample)
        return sample
