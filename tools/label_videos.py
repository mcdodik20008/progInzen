import json

import cv2
import os
import csv
import shutil
from glob import glob
import datetime

FRAMES_DIR = "../clips/frames"
BOOLSHIT_DIR = "../clips_boolshit"
LABELED_DIR = "../clips_labeled"

BOOLSHIT_LOG = "log_boolshit.csv"
LABELS_LOG = "log_clips.csv"
LABELS = {
    ord("a"): "aggressive",
    ord("s"): "suspicious",
    ord("n"): "normal"
}
ENTER_KEY = 13
QUIT_KEY = ord("q")


def label_frame_folders():
    frame_folders = sorted([
        f for f in os.listdir(FRAMES_DIR)
        if os.path.isdir(os.path.join(FRAMES_DIR, f))
    ])

    if os.path.exists(LABELS_LOG):
        with open(LABELS_LOG, "r") as f:
            labeled = {row[0] for row in csv.reader(f)}
    else:
        labeled = set()

    os.makedirs(LABELED_DIR, exist_ok=True)

    with open(LABELS_LOG, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        for folder in frame_folders:
            if folder in labeled:
                continue

            folder_path = os.path.join(FRAMES_DIR, folder)
            frame_paths = sorted(glob(os.path.join(folder_path, "*.jpg")))

            metadata = {}
            metadata_path = os.path.join(folder_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

            if not frame_paths:
                print(f"[!] Пропущен {folder} — нет кадров.")
                continue

            print(f"\nКлип: {folder}")
            print("←/→ — кадры | a/s/n — класс | Enter — сохранить | q — выход")

            idx = 0
            selected_label = None

            while True:
                frame = cv2.imread(frame_paths[idx])
                # Расширим изображение вправо
                h, w = frame.shape[:2]
                display = cv2.copyMakeBorder(frame, 0, 0, 0, 350, cv2.BORDER_CONSTANT, value=(30, 30, 30))

                # Подготовим текст
                info_lines = [
                    f"clip_id: {metadata.get('clip_id', '-')}",
                    f"strategy: {metadata.get('strategy', '-')}",
                    f"label: {selected_label or 'not selected'}",
                    f"frame: {idx + 1} / {len(frame_paths)}",
                    "",
                    "[a/s/n] - mark, [b] - delete",
                    "[o/p] - prev/next",
                    "[Enter] - save, [q] - exit"
                ]

                for i, line in enumerate(info_lines):
                    y = 30 + i * 25
                    cv2.putText(display, line, (w + 10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow("Label Frames", display)
                key = cv2.waitKey(0)

                if key == QUIT_KEY:
                    print("Выход.")
                    cv2.destroyAllWindows()
                    return
                elif key == ord("o") or key == ord("щ"):  # ← O
                    idx = max(0, idx - 1)
                elif key == ord("p") or key == ord("з"):  # → P
                    idx = min(len(frame_paths) - 1, idx + 1)
                elif key in LABELS:
                    selected_label = LABELS[key]
                    print(f"[→] Класс выбран: {selected_label}")
                elif key == ord("b"):
                    print(f"[🗑️] Помечен как мусор: {folder}")
                    os.makedirs(BOOLSHIT_DIR, exist_ok=True)
                    target_path = os.path.join(BOOLSHIT_DIR, folder)
                    shutil.move(folder_path, target_path)
                    timestamp = datetime.datetime.now().isoformat()
                    with open(BOOLSHIT_LOG, "a", newline="") as log_file:
                        log_writer = csv.writer(log_file)
                        log_writer.writerow([folder, "boolshit", timestamp])
                    print(f"[→] Перемещено в {target_path}")
                    break
                elif key == ENTER_KEY and selected_label:
                    timestamp = datetime.datetime.now().isoformat()
                    writer.writerow([folder, selected_label, timestamp])
                    print(f"[✓] {folder} → {selected_label}")
                    target_dir = os.path.join(LABELED_DIR, selected_label)
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.move(folder_path, os.path.join(target_dir, folder))
                    print(f"[→] Перемещено в {target_dir}")
                    break

            cv2.destroyAllWindows()


if __name__ == "__main__":
    label_frame_folders()
