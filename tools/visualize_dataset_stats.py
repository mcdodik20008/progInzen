import csv
import matplotlib.pyplot as plt
from collections import Counter
import os

LABELED_CSV = "log_clips.csv"
BOOLSHIT_CSV = "log_boolshit.csv"

def read_labels(csv_path):
    labels = []
    if not os.path.exists(csv_path):
        print(f"[!] Файл {csv_path} не найден.")
        return labels

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                labels.append(row[1])
    return labels

def main():
    labeled = read_labels(LABELED_CSV)
    trash = read_labels(BOOLSHIT_CSV)

    total_counter = Counter(labeled)
    boolshit_count = len(trash)

    print("[📊] Классы в размеченных данных:")
    for cls, count in total_counter.items():
        print(f"  {cls}: {count}")
    print(f"\n[🗑️] Мусорных клипов: {boolshit_count}")

    # === BAR CHART ===
    plt.figure(figsize=(8, 5))
    plt.bar(total_counter.keys(), total_counter.values())
    plt.title("Количество клипов по классам")
    plt.xlabel("Класс поведения")
    plt.ylabel("Кол-во клипов")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bar_chart.png")
    plt.show()

    # === PIE CHART ===
    total_counter["boolshit"] = boolshit_count
    plt.figure(figsize=(6, 6))
    plt.pie(total_counter.values(), labels=total_counter.keys(), autopct="%1.1f%%", startangle=140)
    plt.title("Распределение данных")
    plt.tight_layout()
    plt.savefig("pie_chart.png")
    plt.show()

if __name__ == "__main__":
    main()
