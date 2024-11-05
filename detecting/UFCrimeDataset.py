import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import re

class UFCrimeDataset(Dataset):
    def __init__(self, root_dir, sequence_length=8, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform if transform else transforms.ToTensor()
        self.data = []
        self.labels = []

        # Регулярное выражение для извлечения номера видео и номера кадра
        pattern = re.compile(r"(\D+)(\d+)_x264_(\d+)\.png")

        # Загрузка всех кадров для каждого класса
        for label_idx, class_dir in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_dir)
            if not os.path.isdir(class_path):
                continue

            # Создаем словарь для хранения кадров каждого видео
            video_frames = {}

            for filename in os.listdir(class_path):
                match = pattern.match(filename)
                if not match:
                    continue

                video_class, video_number, frame_number = match.groups()
                video_number = int(video_number)
                frame_path = os.path.join(class_path, filename)

                if video_number not in video_frames:
                    video_frames[video_number] = []
                video_frames[video_number].append((int(frame_number), frame_path))

            # Сортируем кадры и делим их на последовательности
            for frames in video_frames.values():
                frames.sort()  # Сортируем по номеру кадра
                frames = [frame_path for _, frame_path in frames]

                for i in range(0, len(frames) - sequence_length + 1, sequence_length):
                    sequence = frames[i:i + sequence_length]
                    if len(sequence) == sequence_length:
                        self.data.append(sequence)
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]

        # Загрузка и преобразование каждого кадра в последовательности
        frames = []
        for frame_path in sequence:
            frame = Image.open(frame_path).convert("RGB")
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        frames = torch.stack(frames)  # Размерность будет [sequence_length, C, H, W]
        return frames, label
