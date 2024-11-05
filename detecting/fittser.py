import time
import torch
from typing import List
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from detecting import model_manager
from detecting.UFCrimeDataset import UFCrimeDataset


def fit(
        device: str = "cuda",
        num_epochs: int = 10,
        batch_size: int = 1,
        video_classifier_model: str = "microsoft/xclip-base-patch32",
        labels: List[str] = None,
        learning_rate: float = 1e-4,
):
    if labels is None:
        labels = [
            "Abuse",
            "Arrest",
            "Arson",
            "Assault",
            "Burglary",
            "Explosion",
            "Fighting",
            "RoadAccidents",
            "Robbery",
            "Shooting",
            "Shoplifting",
            "Stealing",
            "Vandalism",
        ]

    # Подготовка устройства и моделей
    device = torch.device(device)
    video_classifier = model_manager.get_classify_model(video_classifier_model, labels, device, False)
    video_classifier.prepare_to_fit()
    video_classifier.model.train().to(device)

    # Загрузчик данных
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = UFCrimeDataset(root_dir="detecting/datasets/UFC-CRIME", transform=transform)  # Замените на путь к вашему датасету
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Оптимизатор и функция потерь
    optimizer = optim.Adam(video_classifier.model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()

        for i, (inputs, inputs_class) in enumerate(dataloader):
            inputs, inputs_class = inputs.to(device), inputs_class.to(device)

            # Прямой проход
            optimizer.zero_grad()
            outputs = video_classifier.model(inputs_class, inputs)
            logits = outputs.logits_per_video

            print("Logits shape:", logits.shape)  # Ожидается [batch_size, num_classes]
            print("Labels shape:", inputs_class.shape)  # Ожидается [batch_size]

            # Вычисление потерь и обратное распространение
            loss = criterion(logits, inputs_class)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # вывод каждые 10 батчей
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0

        end_time = time.time()
        print(f"Epoch {epoch + 1} completed in {end_time - start_time:.2f}s")

    print("Training finished.")
