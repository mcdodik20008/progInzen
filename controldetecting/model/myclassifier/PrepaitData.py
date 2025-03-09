from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split


def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ])
    else:  # для валидации и тестирования
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def load_datasets(train_dir="data/train", val_dir="data/val"):
    train_transform = get_transforms(mode='train')
    val_transform = get_transforms(mode='val')

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    return train_dataset, val_dataset


def get_class_weights(dataset):
    # Подсчёт количества образцов каждого класса
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)

    # Создание весов, обратно пропорциональных частоте класса
    weights = 1. / class_counts
    weights = weights / weights.sum() * len(class_counts)  # нормализация

    # Создание весов для каждого образца
    sample_weights = weights[targets]
    return torch.from_numpy(sample_weights).float()


def create_data_loaders(train_dataset, val_dataset, batch_size=16, num_workers=4, use_weighted_sampling=True):
    # Создаем взвешенный сэмплер для балансировки классов
    if use_weighted_sampling:
        sample_weights = get_class_weights(train_dataset)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def split_dataset(dataset_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Разделяет датасет на обучающую, валидационную и тестовую выборки с сохранением
    соотношения классов.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Пропорции должны суммироваться в 1"

    # Создадим директории для разделенных данных
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    test_dir = os.path.join(dataset_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Для каждого класса создаем папки в train, val и test
    classes = [d for d in os.listdir(dataset_dir)
               if os.path.isdir(os.path.join(dataset_dir, d)) and
               d not in ['train', 'val', 'test']]

    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        # Получаем все файлы для текущего класса
        class_dir = os.path.join(dataset_dir, cls)
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        # Делим на train и temporary sets
        train_files, temp_files = train_test_split(
            files, train_size=train_ratio, random_state=seed, shuffle=True
        )

        # Делим temporary на val и test sets
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_files, test_files = train_test_split(
            temp_files, train_size=val_ratio_adjusted, random_state=seed, shuffle=True
        )

        # Копируем файлы в соответствующие директории
        import shutil
        for f in train_files:
            shutil.copy2(os.path.join(class_dir, f), os.path.join(train_dir, cls, f))
        for f in val_files:
            shutil.copy2(os.path.join(class_dir, f), os.path.join(val_dir, cls, f))
        for f in test_files:
            shutil.copy2(os.path.join(class_dir, f), os.path.join(test_dir, cls, f))

    return train_dir, val_dir, os.path.join(dataset_dir, 'test')