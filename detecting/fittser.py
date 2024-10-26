import os

import torch
import torch.nn as nn
import torch.optim as optim

from detecting import model_manager
from detecting.dataset_manager import DatasetManager


def fit(dataset_manager: DatasetManager, video_classifier_model, device ="cuda", fp16=False, num_epochs=10, learning_rate=1e-4):
    """
    Args:
        dataset_loader - типа загрузчик дата сетов, все делает специальный модуль
        video_classifier_model - название модели для классификации на хагинфасе или путь до нее в жд
        device - на чем будем обучать
        fp16 - пока хз зачем
    """

    imported_model = model_manager.get_classify_model(video_classifier_model, dataset_manager.get_labels(), device, fp16)
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

            # Изменяем форму из [N, C, H, W] в [N, H, W, C]
            images = images.permute(0, 2, 3, 1).cpu().detach().numpy()  # Теперь у нас [N, H, W, C]

            # Преобразуем список изображений
            crops = [img for img in images]  # Каждый элемент уже в формате [H, W, C]

            # Проверяем форму каждого crop
            for idx, crop in enumerate(crops):
                if crop.ndim != 3 or crop.shape[2] != 3:
                    print(f"Invalid shape for crop {idx}: {crop.shape}")

            print(f"Type of crops: {type(crops)}")
            print(f"Number of crops: {len(crops)}")

            # Передаем crops в preprocess_crops_for_video_cls
            processed_images = imported_model.preprocess_crops_for_video_cls(crops)  # Предобработка изображений

            optimizer.zero_grad()  # Обнуляем градиенты
            outputs = model(processed_images)  # Прямой проход
            loss = criterion(outputs, labels)  # Вычисление потерь
            loss.backward()  # Обратный проход
            optimizer.step()  # Обновление весов
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / dataset_manager.get_num_classes():.4f}')

    # Сохранение модели после обучения
    model_path = "detecting/models/fitted/" + os.path.basename(video_classifier_model)
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')