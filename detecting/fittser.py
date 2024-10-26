import os

import torch
import torch.nn as nn
import torch.optim as optim

from detecting import model_manager
from detecting.dataset_manager import DatasetManager


def fit(dataset_manager: DatasetManager, video_classifier_model, device="cuda", fp16=False, num_epochs=10,
        learning_rate=1e-4):
    """
    Args:
        dataset_manager: Типа загрузчик дата сетов, все делает специальный модуль.
        video_classifier_model: Название модели для классификации на Hugging Face или путь до нее в ЖД.
        device: На чем будем обучать.
        fp16: Использовать ли fp16 для обучения.
    """

    imported_model = model_manager.get_classify_model(video_classifier_model, dataset_manager.get_labels(), device,fp16)
    model = imported_model.prepare_to_fit()

    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Обучение модели
    model.train()  # Переключаем модель в режим обучения
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in dataset_manager.get_loader():  # Итерируемся по DataLoader
            images = images.to(device)
            labels = labels.to(device)

            # Проверяем размерность images
            print(f"Shape of images batch: {images.shape}")  # Например, [N, C, H, W]

            # Изменяем форму из [N, C, H, W] в [N, H, W, C]
            crops = [img.permute(1, 2, 0).cpu().detach().numpy() for img in images]  # Теперь у нас [H, W, C]

            # Проверяем форму каждого crop
            for idx, crop in enumerate(crops):
                if crop.ndim != 3 or crop.shape[2] != 3:
                    print(f"Invalid shape for crop {idx}: {crop.shape}")



            # Передаем crops в preprocess_crops_for_video_cls
            processed_images = imported_model.preprocess_crops_for_video_cls(crops)  # Предобработка изображений
            processed_images = processed_images.permute(1, 0, 2, 4, 3)

            # Создание attention_mask с нужной размерностью [batch_size, sequence_length]
            batch_size, seq_len = processed_images.shape[0], processed_images.shape[1]
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.float32).to(device)

            # Добавляем проверку на None
            if processed_images is None:
                print("Error: processed_images is None after preprocessing!")
                return

            optimizer.zero_grad()  # Обнуляем градиенты
            # Используем автоматическое смешанное обучение, если включено
            with torch.cuda.amp.autocast(fp16):
                print(f"Shape of processed_images before model call: {processed_images.shape}")
                outputs = model(labels, processed_images, attention_mask=attention_mask)  # Прямой проход
                loss = criterion(outputs, labels)  # Вычисление потерь

            loss.backward()  # Обратный проход
            optimizer.step()  # Обновление весов

            total_loss += loss.item()  # Суммируем потери

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataset_manager.get_loader()):.4f}')

    # Сохранение модели после обучения
    model_path = "detecting/models/fitted/" + os.path.basename(video_classifier_model)
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
