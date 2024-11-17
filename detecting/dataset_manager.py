import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset?resource=download-directory
class DatasetManager:
    def __init__(self, dataset_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.__dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    def get_dataset(self):
        return self.__dataset

    def get_num_classes(self):
        return len(self.__dataset.classes)

    def get_labels(self):
        if len(self.__dataset) == 0:
            return [
                "walking",
                "running",
                "brushing teeth",
                "looking into phone",
                "weight lifting",
                "cooking",
                "sitting",
            ]

        val = np.array(self.__dataset.classes).tolist()
        val = val + [
                "walking",
                "running",
                "brushing teeth",
                "looking into phone",
                "weight lifting",
                "cooking",
                "sitting",
            ]

        return val

    def get_loader(self):
        return DataLoader(self.__dataset, batch_size=32, shuffle=True)