# VideoProcessor.py
import cv2
import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
from collections import deque


class VideoSequenceExtractor:
    def __init__(self, sequence_length=16, frame_step=2, resize=(224, 224)):
        self.sequence_length = sequence_length
        self.frame_step = frame_step
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_sequences(self, video_path, output_dir=None, overlap=0.5):
        """Извлекает последовательности кадров из видео с заданным перекрытием"""
        cap = cv2.VideoCapture(video_path)
        frame_buffer = deque(maxlen=self.sequence_length)
        frame_count = 0
        sequence_count = 0
        sequences = []

        # Шаг для перекрытия последовательностей
        step = int(self.sequence_length * (1 - overlap))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_step == 0:
                # Преобразование в RGB и нормализация
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_transformed = self.transform(frame_pil)

                frame_buffer.append(frame_transformed)

                # Когда буфер заполнен, создаем последовательность
                if len(frame_buffer) == self.sequence_length and (len(frame_buffer) == self.sequence_length and
                                                                  frame_count % step == 0):
                    sequence = torch.stack(list(frame_buffer))  # [T, C, H, W]

                    if output_dir:
                        sequence_dir = os.path.join(output_dir, f"sequence_{sequence_count}")
                        os.makedirs(sequence_dir, exist_ok=True)

                        # Сохраняем отдельные кадры для отладки
                        for i, frame_tensor in enumerate(frame_buffer):
                            frame_np = (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(sequence_dir, f"frame_{i:03d}.jpg"),
                                        cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

                    sequences.append(sequence)
                    sequence_count += 1

            frame_count += 1

        cap.release()
        return sequences if sequences else None

    def extract_sequences_batch(self, video_dir, category, output_base_dir=None):
        """Извлекает последовательности из всех видео в директории и сортирует по категориям"""
        all_sequences = []
        all_labels = []

        # Получаем числовую метку для категории
        category_labels = {"нормальное": 0, "подозрительное": 1, "опасное": 2}
        label = category_labels.get(category.lower(), -1)

        if label == -1:
            raise ValueError(f"Неизвестная категория: {category}")

        for video_file in os.listdir(video_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(video_dir, video_file)

                if output_base_dir:
                    output_dir = os.path.join(output_base_dir, category, os.path.splitext(video_file)[0])
                    os.makedirs(output_dir, exist_ok=True)
                else:
                    output_dir = None

                sequences = self.extract_sequences(video_path, output_dir)

                if sequences:
                    all_sequences.extend(sequences)
                    all_labels.extend([label] * len(sequences))

        return all_sequences, all_labels