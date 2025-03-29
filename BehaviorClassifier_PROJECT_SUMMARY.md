
# 🧠 BehaviorClassifier — Project Summary

## 🧠 Обзор

BehaviorClassifier — это проект по определению поведения человека (агрессивное, подозрительное, нормальное) на видео с использованием модели ViViT.  
Система работает как в пакетном режиме, так и в реальном времени, используя YOLO для детекции и трекинга, классификатор поведения, Colab для обучения и мощную разметку данных с логами и стратегиями.

---

## 📦 Компоненты

- [`AppController`](core/controller.py): запускает обработку видео, сохраняет фреймы  
- [`TrackClipBuffer`](core/track_clip_buffer.py): собирает 16 кадров по `track_id`  
- [`FrameProcessor`](core/frame_processor.py): использует YOLO для трекинга  
- [`label_frame_folders.py`](tools/label_frame_folders.py): инструмент ручной разметки с горячими клавишами  
- [`convert_dataset.py`](tools/convert_dataset.py): создаёт версионные `.npz` датасеты  
- [`BehaviorClassifier`](classifier/behavior_classifier.py): одиночный и batch-инференс  
- [`LiveAnalyzer`](core/live_analyzer.py): анализ видео в реальном времени с отрисовкой  
- [`sync_with_colab.py`](tools/sync_with_colab.py): отправка датасета на Google Drive  
- [`ViViT_Trainer_Colab.ipynb`](colab/ViViT_Trainer_Colab.ipynb): обучение модели с отчётом


---

## ☁️ Colab + Drive интеграция

- Скрипт `sync_with_colab.py` загружает актуальный датасет в `/datasets/vN.npz`
- Google Colab автоматически загружает модель и обучает её
- Результаты сохраняются в `/models/vN/`:  
  - `model.pth`  
  - `report.md`  
  - `confusion_matrix.png`

---

## 📊 Метрики и отчётность

- Используются метрики:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Генерируется **markdown-отчёт** с таблицей метрик и анализом:
  > Например, низкий F1 может означать либо плохой recall, либо низкий precision.

---

## 🎥 LiveAnalyzer

Компонент `LiveAnalyzer`:
- получает потоковое видео
- детектирует людей (YOLO + трекинг)
- накапливает кадры по `track_id`
- классифицирует поведение через `BehaviorClassifier`
- отрисовывает результат: `bbox`, label и вероятности на экране

---

## 📚 README

Проект снабжён README.md, описывающим шаги от установки до обучения модели и запуска `LiveAnalyzer`.  
README содержит шутки, подсказки и ориентирован на удобство новых пользователей.

---

## 🧠 Заключение

Проект завершён на высоком уровне: есть трекинг, буферы, стратегия сохранения кадров, мощная ручная разметка, логирование, обучение модели, Colab-интеграция, отчёты и real-time анализ.  
Система легко расширяется под тревожные алерты, Telegram-ботов, REST API и интеграцию в реальную городскую инфраструктуру.

---
