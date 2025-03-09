# PrepareVideoData.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from VideoProcessor import VideoSequenceExtractor
from VideoDataset import BehaviorVideoDataset
from torchvision import transforms
import shutil


def prepare_video_datasets(
        video_root_dir,
        output_dir,
        sequence_length=16,
        frame_step=2,
        val_ratio=0.2,
        batch_size=8,
        num_workers=4,
        resize=(224, 224)
):
    # Создаем директории для сохранения последовательностей
    os.makedirs(output_dir, exist_ok=True)

    # Словарь для категориальных меток
    category_map = {
        "нормальное": 0,
        "подозрительное": 1,
        "опасное": 2
    }

    # Инициализируем экстрактор последовательностей
    extractor = VideoSequenceExtractor(
        sequence_length=sequence_length,
        frame_step=frame_step,
        resize=resize
    )

    all_sequences = []
    all_labels = []

    # Обрабатываем видео по категориям
    for category_name in os.listdir(video_root_dir):
        category_dir = os.path.join(video_root_dir, category_name)
        if os.path.isdir(category_dir) and category_name.lower() in category_map:
            category_label = category_map[category_name.lower()]

            print(f"Обработка категории: {category_name} (метка: {category_label})")

            # Проверяем наличие кэшированных данных
            cache_file = os.path.join(output_dir, f"{category_name}_sequences.pt")
            if os.path.exists(cache_file):
                print(f"Загружаем кэшированные последовательности из {cache_file}")
                cached_data = torch.load(cache_file)
                category_sequences = cached_data["sequences"]
                category_labels = cached_data["labels"]
            else:
                # Извлекаем последовательности из видео
                category_sequences = []
                category_labels = []

                for video_file in os.listdir(category_dir):
                    if video_file.endswith(('.mp4', '.avi', '.mov')):
                        video_path = os.path.join(category_dir, video_file)
                        print(f"  Обработка видео: {video_file}")

                        # Извлекаем последовательности
                        sequences = extractor.extract_sequences(
                            video_path,
                            output_dir=os.path.join(output_dir, f"{category_name}_{os.path.splitext(video_file)[0]}")
                        )

                        if sequences:
                            category_sequences.extend(sequences)
                            category_labels.extend([category_label] * len(sequences))

                # Кэшируем данные
                if category_sequences:
                    torch.save({
                        "sequences": category_sequences,
                        "labels": category_labels
                    }, cache_file)

            all_sequences.extend(category_sequences)
            all_labels.extend(category_labels)

    # Информация о наборе данных
    print(f"\nОбщее количество последовательностей: {len(all_sequences)}")
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        for cat_name, cat_id in category_map.items():
            if cat_id == label:
                print(f"  {cat_name}: {count} последовательностей")

    # Создаем наборы данных
    dataset = BehaviorVideoDataset(all_sequences, all_labels, mode='train')

    # Разделение на train и val
    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Применяем трансформации к обучающему набору
    train_dataset.dataset.mode = 'train'
    val_dataset.dataset.mode = 'val'

    # Создаем загрузчики данных
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

    return train_loader, val_loader, category_map