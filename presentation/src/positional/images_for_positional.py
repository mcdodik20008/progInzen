import numpy as np
import matplotlib.pyplot as plt
import os

tokens = ["A", "black", "cat"]
embedding_vectors = np.array([[0.2, 0.5], [0.1, 0.6], [0.4, 0.9]])  # Векторы встраивания
position_vectors = np.array([[1, 0], [2, 0], [3, 0]])  # Позиционные векторы
result_vectors = embedding_vectors + position_vectors  # Итоговые векторы

output_dir = "png_images"
os.makedirs(output_dir, exist_ok=True)


def save_image(frame):
    fig, ax = plt.subplots()

    ax.set_xlim(-0.1, 4)
    ax.set_ylim(-0.1, 4)

    ax.plot([0, embedding_vectors[frame, 0]], [0, embedding_vectors[frame, 1]], 'b-', label="Вектор встраивания")
    ax.plot([0, position_vectors[frame, 0]], [0, position_vectors[frame, 1]], 'g-', label="Позиционный вектор")
    ax.plot([0, result_vectors[frame, 0]], [0, result_vectors[frame, 1]], 'r-', label="Итоговый вектор")

    ax.text(1, 3.5, f"Токен: {tokens[frame]}", ha="center", fontsize=12)
    ax.text(1, 3.2, f"Вектор встраивания: {embedding_vectors[frame]}", ha="center", fontsize=12)
    ax.text(1, 2.9, f"Позиционный вектор: {position_vectors[frame]}", ha="center", fontsize=12)
    ax.text(1, 2.6, f"Итоговый вектор: {result_vectors[frame]}", ha="center", fontsize=12)

    ax.legend(loc='upper right', fontsize=10)

    image_path = os.path.join(output_dir, f"step_{frame + 1}.png")
    fig.savefig(image_path)
    plt.close(fig)
    return image_path

image_paths = [save_image(i) for i in range(len(tokens))]
image_paths
