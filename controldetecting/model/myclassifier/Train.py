import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def get_criterion_optimizer(model, lr=2e-5, weight_decay=1e-4):
    # Добавляем веса для несбалансированных классов
    # Веса можно настроить в зависимости от распределения данных
    # class_weights = torch.tensor([1.0, 2.0, 3.0]).to(get_device())
    criterion = nn.CrossEntropyLoss()

    # Используем AdamW с весовым затуханием для регуляризации
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Планировщик скорости обучения
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    return criterion, optimizer, scheduler


def train_epoch(model, loader, criterion, optimizer, device, accumulation_steps=1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()  # Обнуляем градиенты перед началом эпохи

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Нормализация потерь при накоплении градиентов
        loss = loss / accumulation_steps
        loss.backward()

        # Обновляем веса только после накопления градиентов
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        # Вычисляем точность на тренировочном наборе
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Проверяем, не остались ли ненакопленные градиенты
    if (len(loader) % accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad()

    accuracy = correct / total
    return total_loss / len(loader), accuracy


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total

    # Вычисляем матрицу путаницы и отчет о классификации
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)

    return total_loss / len(loader), accuracy, cm, report


def save_model(model, optimizer, epoch, loss, accuracy, path="models/vivit_classifier.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }, path)