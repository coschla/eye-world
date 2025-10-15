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
        img, label = sample
        img = self.transform(img)
        # TODO: Label is returned as a list. Return a tensor.
        return img, label[-1]


class ComposePreprocessor:
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def __call__(self, sample):
        for p in self.preprocessors:
            sample = p(sample)
        return sample
