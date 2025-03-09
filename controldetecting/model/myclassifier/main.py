# main_video.py
import os
from datetime import datetime

import torch
import yaml

from PrepareVIdeoData import prepare_video_datasets
from Train import get_criterion_optimizer, validate, save_model
from ViViTModelVideo import ViViTVideoClassifier, get_device


def train_epoch_video(model, loader, criterion, optimizer, device, accumulation_steps=1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for i, (sequences, labels) in enumerate(loader):
        # Последовательности имеют форму [B, T, C, H, W]
        sequences, labels = sequences.to(device), labels.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, labels)

        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    if (len(loader) % accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad()

    accuracy = correct / total
    return total_loss / len(loader), accuracy


def main():
    # Создаем директорию для эксперимента
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/video_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)

    # Настройки эксперимента
    config = {
        "dataset": {
            "video_root_dir": "data/videos",  # Корневая директория с видео по категориям
            "output_dir": "data/processed_video_sequences",
            "sequence_length": 16,  # Длина последовательностей (кадров)
            "frame_step": 2,  # Шаг между кадрами
            "val_ratio": 0.2,  # Доля валидационных данных
            "batch_size": 4  # Меньший размер батча для видео
        },
        "model": {
            "backbone": "vit",  # "vit", "resnet", "efficientnet"
            "num_classes": 3,  # нормальное, подозрительное, опасное
            "pretrained": True,
            "dropout_rate": 0.5
        },
        "training": {
            "initial_epochs": 5,  # Эпохи для обучения только классификатора
            "fine_tuning_epochs": 15,  # Эпохи для дообучения всей модели
            "lr_initial": 5e-5,
            "lr_fine_tuning": 1e-6,
            "weight_decay": 1e-4,
            "patience": 7,
            "accumulation_steps": 4  # Больше шагов накопления для компенсации маленького батча
        }
    }

    # Сохраняем конфигурацию
    with open(f"{experiment_dir}/config.yaml", 'w') as f:
        yaml.dump(config, f)

    # Параметры из конфига
    VIDEO_ROOT_DIR = config["dataset"]["video_root_dir"]
    OUTPUT_DIR = config["dataset"]["output_dir"]
    SEQUENCE_LENGTH = config["dataset"]["sequence_length"]
    FRAME_STEP = config["dataset"]["frame_step"]
    VAL_RATIO = config["dataset"]["val_ratio"]
    BATCH_SIZE = config["dataset"]["batch_size"]
    NUM_CLASSES = config["model"]["num_classes"]
    INITIAL_EPOCHS = config["training"]["initial_epochs"]
    FINE_TUNING_EPOCHS = config["training"]["fine_tuning_epochs"]
    TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNING_EPOCHS
    MODEL_PATH = f"{experiment_dir}/best_model_video.pth"

    # Журнал для отслеживания результатов
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    # Устройство
    device = get_device()
    print(f"Используется устройство: {device}")

    # Подготовка данных
    train_loader, val_loader, category_map = prepare_video_datasets(
        VIDEO_ROOT_DIR,
        OUTPUT_DIR,
        sequence_length=SEQUENCE_LENGTH,
        frame_step=FRAME_STEP,
        val_ratio=VAL_RATIO,
        batch_size=BATCH_SIZE
    )

    print("Категории:", category_map)

    # Построение модели для видео
    model = ViViTVideoClassifier(
        num_classes=NUM_CLASSES,
        pretrained=config["model"]["pretrained"],
        backbone=config["model"]["backbone"],
        dropout_rate=config["model"]["dropout_rate"]
    ).to(device)

    # Этап 1: Заморозка основы и обучение только классификатора
    model.freeze_backbone()
    print("Заморожена основа модели для первичного обучения")

    criterion, optimizer, scheduler = get_criterion_optimizer(
        model,
        lr=config["training"]["lr_initial"],
        weight_decay=config["training"]["weight_decay"]
    )

    # Параметры для раннего останова
    best_val_accuracy = 0.0
    patience_counter = 0

    # Фаза 1: Обучение только классификатора
    print("Фаза 1: Обучение только классификатора...")
    for epoch in range(INITIAL_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        train_loss, train_acc = train_epoch_video(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            accumulation_steps=config["training"]["accumulation_steps"]
        )

        # Для валидации используем обычную функцию validate
        val_loss, val_acc, cm, report = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Эпоха {epoch + 1}/{INITIAL_EPOCHS}: "
              f"Train Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.4f}, "
              f"LR={current_lr:.1e}")

        # Сохраняем лучшую модель
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_model(model, optimizer, epoch, val_loss, val_acc, MODEL_PATH)
            print(f"Сохранена лучшая модель с точностью {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

    # Фаза 2: Дообучение всей модели
    model.unfreeze_backbone()
    print("\nФаза 2: Дообучение всей модели...")

    criterion, optimizer, scheduler = get_criterion_optimizer(
        model,
        lr=config["training"]["lr_fine_tuning"],
        weight_decay=config["training"]["weight_decay"]
    )

    patience_counter = 0

    # Обучение (вторая фаза)
    for epoch in range(INITIAL_EPOCHS, TOTAL_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        train_loss, train_acc = train_epoch_video(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            accumulation_steps=config["training"]["accumulation_steps"]
        )

        val_loss, val_acc, cm, report = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Эпоха {epoch + 1}/{TOTAL_EPOCHS}: "
              f"Train Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.4f}, "
              f"LR={current_lr:.1e}")

        # Сохраняем лучшую модель
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_model(model, optimizer, epoch, val_loss, val_acc, MODEL_PATH)
            print(f"Сохранена лучшая модель с точностью {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

            # Проверка на раннюю остановку
            if patience_counter >= config["training"]["patience"]:
                print(f"Раннее прекращение обучения: нет улучшений в течение {patience_counter} эпох")
                break

        # Сохраняем историю обучения
    import pandas as pd
    import matplotlib.pyplot as plt

    # Преобразуем историю в DataFrame для удобства
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"{experiment_dir}/training_history.csv", index=False)

    # График потерь и точности
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{experiment_dir}/training_curves.png")

    # Загружаем лучшую модель для итоговой оценки
    best_model_state = torch.load(MODEL_PATH)
    model.load_state_dict(best_model_state['model_state_dict'])

    # Финальная оценка на валидационном наборе
    val_loss, val_acc, confusion_matrix, classification_rep = validate(model, val_loader, criterion, device)

    print("\nРезультаты лучшей модели:")
    print(f"Валидационная точность: {val_acc:.4f}")
    print(f"Матрица путаницы:\n{confusion_matrix}")

    # Сохраняем отчет о классификации
    pd.DataFrame(classification_rep).transpose().to_csv(f"{experiment_dir}/classification_report.csv")

    print(f"\nОбучение завершено! Результаты сохранены в {experiment_dir}")


if __name__ == "__main__":
    main()