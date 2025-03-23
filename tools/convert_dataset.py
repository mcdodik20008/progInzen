import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
import re

CLIPS_ROOT = "../clips_labeled"
DATASETS_DIR = "../ready_datasets"
NUM_FRAMES = 16
FRAME_SIZE = (224, 224)
LABEL_MAP = {
    "aggressive": 0,
    "suspicious": 1,
    "normal": 2
}


def get_next_version_number() -> int:
    os.makedirs(DATASETS_DIR, exist_ok=True)
    versions = [
        int(re.findall(r"v(\d+)\.npz", f)[0])
        for f in os.listdir(DATASETS_DIR)
        if f.startswith("v") and f.endswith(".npz")
    ]
    return max(versions) + 1 if versions else 1


def read_clip_from_frames(folder_path: str, num_frames: int, resize_to: tuple) -> np.ndarray:
    frame_paths = sorted(glob(os.path.join(folder_path, "frame_*.jpg")))
    if not frame_paths:
        raise ValueError(f"Нет кадров в папке: {folder_path}")

    total = len(frame_paths)
    idxs = np.linspace(0, total - 1, num_frames).astype(int)

    frames = []
    for i in idxs:
        frame = cv2.imread(frame_paths[i])
        if frame is None:
            raise ValueError(f"Не удалось прочитать кадр: {frame_paths[i]}")
        resized = cv2.resize(frame, resize_to)
        frames.append(resized)

    return np.array(frames)


def convert_dataset():
    version = get_next_version_number()
    output_file = os.path.join(DATASETS_DIR, f"v{version}.npz")

    X = []
    y = []

    for label_name in LABEL_MAP.keys():
        label_dir = os.path.join(CLIPS_ROOT, label_name)
        if not os.path.isdir(label_dir):
            continue

        clip_folders = sorted([
            os.path.join(label_dir, f) for f in os.listdir(label_dir)
            if os.path.isdir(os.path.join(label_dir, f))
        ])

        for clip_path in tqdm(clip_folders, desc=f"Processing {label_name}"):
            try:
                frames = read_clip_from_frames(clip_path, NUM_FRAMES, FRAME_SIZE)
                X.append(frames)
                y.append(LABEL_MAP[label_name])
            except Exception as e:
                print(f"[!] Пропущен {clip_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"[✓] Сохранено: {X.shape[0]} клипов, размер одного: {X.shape[1:]}")
    np.savez_compressed(output_file, X=X, y=y)
    print(f"[→] Датасет сохранён в: {output_file}")


if __name__ == "__main__":
    convert_dataset()
