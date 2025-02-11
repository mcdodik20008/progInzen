# https://habr.com/ru/articles/827474/

from transformers import VivitForVideoClassification, VivitImageProcessor
import torch
import cv2

# Загружаем модель и процессинг кадров
model_name = "google/vivit-b-16x2-kinetics400"
processor = VivitImageProcessor.from_pretrained(model_name)
model = VivitForVideoClassification.from_pretrained(model_name)

# Функция для загрузки и обработки видео
def load_video(video_path, num_frames=32, frame_height=480, frame_width=480):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_width, frame_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if len(frames) < num_frames:
        raise ValueError(f"Video is too short. Needed: {num_frames}, Found: {len(frames)}")
    return frames

# Функция для загрузки меток
def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = f.read().splitlines()
    return labels


# Функция для получения предсказаний
def predict_label(video_path, label_map_path):
    frames = load_video(video_path)
    inputs = processor(images=frames, return_tensors="pt")
    outputs = model(**inputs)
    # Загрузка меток
    kinetics_labels = load_labels(label_map_path)
    # Добавим вывод вероятностей
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    top_5_probs, top_5_indices = torch.topk(probabilities[0], 5)
    top_5_predictions = [kinetics_labels[idx.item()] for idx in top_5_indices]

    print("Топ 5 предсказаний:")
    for i, (label, prob) in enumerate(zip(top_5_predictions, top_5_probs), 1):
        print(f"{i}. {label}: {prob.item() * 100:.2f}%")

# Путь к видеофайлу
video_path = 'snow.mp4'

# Загрузка label_map.txt происходит по след. ссылке: https://github.com/google-deepmind/kinetics-i3d/blob/master/data/label_map.txt
# Получение предсказаний
predict_label(video_path, "label_map.txt")