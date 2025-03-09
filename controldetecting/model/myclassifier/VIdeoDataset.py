# VideoDataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import random


class BehaviorVideoDataset(Dataset):
    def __init__(self, sequences, labels=None, transform=None,
                 sequence_length=16, mode='train', augment_prob=0.5):
        self.sequences = sequences  # список путей к последовательностям или сами последовательности
        self.labels = labels
        self.transform = transform
        self.sequence_length = sequence_length
        self.mode = mode
        self.augment_prob = augment_prob

        # Базовая трансформация
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Аугментация для видео
        self.video_augment = VideoAugmentation(p=augment_prob)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]  # [T, C, H, W]

        # Применяем временную аугментацию
        if self.mode == 'train' and random.random() < self.augment_prob:
            sequence = self.video_augment(sequence)

        # Применяем трансформацию если она указана
        if self.transform:
            # Apply transforms to each frame
            transformed_seq = []
            for frame in sequence:
                # Convert tensor to PIL image if necessary
                if isinstance(frame, torch.Tensor):
                    frame = transforms.ToPILImage()(frame)
                transformed_frame = self.transform(frame)
                transformed_seq.append(transformed_frame)
            sequence = torch.stack(transformed_seq)

        if self.labels is not None:
            label = self.labels[idx]
            return sequence, label
        else:
            return sequence


class VideoAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video_tensor):
        # video_tensor: [T, C, H, W]
        if random.random() < self.p:
            # Временное отражение (реверс последовательности)
            if random.random() < 0.5:
                video_tensor = torch.flip(video_tensor, [0])

            # Временная интерполяция (замедление или ускорение)
            if random.random() < 0.5:
                factor = random.uniform(0.8, 1.2)
                T, C, H, W = video_tensor.shape
                new_T = max(1, int(T * factor))
                if new_T != T:
                    indices = torch.linspace(0, T - 1, new_T)
                    indices = indices.round().long().clamp(0, T - 1)
                    video_tensor = video_tensor[indices]

            # Временной сдвиг (удаление начальных/конечных кадров)
            if random.random() < 0.5:
                T = video_tensor.shape[0]
                max_shift = max(1, int(T * 0.2))
                shift = random.randint(0, max_shift)
                if random.random() < 0.5:  # сдвиг начала
                    video_tensor = video_tensor[shift:]
                else:  # сдвиг конца
                    video_tensor = video_tensor[:-shift]

                # Заполняем до исходной длины
                if video_tensor.shape[0] < T:
                    padding = T - video_tensor.shape[0]
                    if random.random() < 0.5:
                        # Дополняем начало повторением первого кадра
                        pad_frames = video_tensor[0:1].repeat(padding, 1, 1, 1)
                        video_tensor = torch.cat([pad_frames, video_tensor], dim=0)
                    else:
                        # Дополняем конец повторением последнего кадра
                        pad_frames = video_tensor[-1:].repeat(padding, 1, 1, 1)
                        video_tensor = torch.cat([video_tensor, pad_frames], dim=0)

        return video_tensor