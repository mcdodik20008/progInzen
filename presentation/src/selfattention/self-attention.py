import numpy as np
import matplotlib.pyplot as plt
import os

# Токены и их запросы, ключи и значения
tokens = ["A", "black", "cat"]
query_vectors = np.array([[0.2, 0.5], [0.1, 0.6], [0.4, -0.2]])  # Запросы (Q)
key_vectors = np.array([[0.3, 0.7], [0.2, 0.4], [0.5, -0.1]])  # Ключи (K)
value_vectors = np.array([[0.1, 0.2], [0.3, 0.5], [0.6, 0.1]])  # Значения (V)

# Создаем директорию для хранения изображений
output_dir = "png_images"
os.makedirs(output_dir, exist_ok=True)


def save_step_image(frame, step, query_vectors, key_vectors, value_vectors, attention_weights, weighted_values):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Устанавливаем диапазон осей от -1 до 2 для X и Y
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)

    # Токены для отображения
    token = tokens[frame]

    if step == 1:  # Этап 1: Инициализация
        # Рисуем запрос, ключ и значение для токена
        ax.plot([0, query_vectors[frame, 0]], [0, query_vectors[frame, 1]], 'bo-', label="Запрос (Q)")
        ax.plot([0, key_vectors[frame, 0]], [0, key_vectors[frame, 1]], 'go-', label="Ключ (K)")
        ax.plot([0, value_vectors[frame, 0]], [0, value_vectors[frame, 1]], 'ro-', label="Значение (V)")

        ax.text(query_vectors[frame, 0], query_vectors[frame, 1], f"Q_{token}", ha="center", fontsize=12)
        ax.text(key_vectors[frame, 0], key_vectors[frame, 1], f"K_{token}", ha="center", fontsize=12)
        ax.text(value_vectors[frame, 0], value_vectors[frame, 1], f"V_{token}", ha="center", fontsize=12)

    elif step == 2:  # Этап 2: Скалярное произведение (Q * K^T)
        # Вычисляем скалярные произведения
        scalar_products = np.dot(query_vectors[frame], key_vectors.T)

        # Рисуем вектор запроса и ключа для токена black
        ax.plot([0, query_vectors[frame, 0]], [0, query_vectors[frame, 1]], 'bo-', label="Запрос (Q)")
        ax.plot([0, key_vectors[frame, 0]], [0, key_vectors[frame, 1]], 'go-', label="Ключ (K)")

        ax.text(query_vectors[frame, 0], query_vectors[frame, 1], f"Q_{token}", ha="center", fontsize=12)
        ax.text(key_vectors[frame, 0], key_vectors[frame, 1], f"K_{token}", ha="center", fontsize=12)

        # Скалярные произведения (S) для визуализации
        for i, scalar_product in enumerate(scalar_products):
            ax.text(0.5, 0.5 + 0.2 * i, f"S_{tokens[i]} = {scalar_product:.2f}", fontsize=12)

    elif step == 3:  # Этап 3: Применение Softmax
        # Вычисляем скалярные произведения
        scalar_products = np.dot(query_vectors[frame], key_vectors.T)
        # Применяем softmax для вычисления весов внимания
        softmax_weights = np.exp(scalar_products) / np.sum(np.exp(scalar_products))

        # Рисуем веса внимания
        ax.plot([0, query_vectors[frame, 0]], [0, query_vectors[frame, 1]], 'bo-', label="Запрос (Q)")
        ax.plot([0, key_vectors[frame, 0]], [0, key_vectors[frame, 1]], 'go-', label="Ключ (K)")
        ax.plot([0, value_vectors[frame, 0]], [0, value_vectors[frame, 1]], 'ro-', label="Значение (V)")

        ax.text(query_vectors[frame, 0], query_vectors[frame, 1], f"Q_{token}", ha="center", fontsize=12)
        ax.text(key_vectors[frame, 0], key_vectors[frame, 1], f"K_{token}", ha="center", fontsize=12)
        ax.text(value_vectors[frame, 0], value_vectors[frame, 1], f"V_{token}", ha="center", fontsize=12)

        # Веса внимания
        for i, weight in enumerate(softmax_weights):
            ax.text(0.5, 0.5 + 0.2 * i, f"Attention Weight_{tokens[i]} = {weight:.2f}", fontsize=12)

    elif step == 4:  # Этап 4: Итоговый вектор (V * Attention Weights)
        # Вычисляем скалярные произведения
        scalar_products = np.dot(query_vectors[frame], key_vectors.T)
        # Применяем softmax для вычисления весов внимания
        softmax_weights = np.exp(scalar_products) / np.sum(np.exp(scalar_products))
        # Итоговый вектор = V * Attention Weights
        weighted_values = np.dot(softmax_weights, value_vectors)  # Сумма V * Attention Weights

        # Рисуем итоговый вектор
        ax.plot([0, weighted_values[0]], [0, weighted_values[1]], 'mo-', label="Итоговый вектор (V * внимание)")

        # Рисуем вектора запроса, ключа и значения для токена
        ax.plot([0, query_vectors[frame, 0]], [0, query_vectors[frame, 1]], 'bo-', label="Запрос (Q)")
        ax.plot([0, key_vectors[frame, 0]], [0, key_vectors[frame, 1]], 'go-', label="Ключ (K)")
        ax.plot([0, value_vectors[frame, 0]], [0, value_vectors[frame, 1]], 'ro-', label="Значение (V)")

        ax.text(query_vectors[frame, 0], query_vectors[frame, 1], f"Q_{token}", ha="center", fontsize=12)
        ax.text(key_vectors[frame, 0], key_vectors[frame, 1], f"K_{token}", ha="center", fontsize=12)
        ax.text(value_vectors[frame, 0], value_vectors[frame, 1], f"V_{token}", ha="center", fontsize=12)

        # Аннотируем итоговый вектор
        ax.text(weighted_values[0], weighted_values[1], f"Итоговый вектор", ha="center", fontsize=12)

    # Легенда
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title(f"Шаг {step}: Механизм Self-Attention для токена {token}")

    # Сохраняем изображение как PNG
    image_path = os.path.join(output_dir, f"attention_step_{frame + 1}_step_{step}.png")
    fig.savefig(image_path)
    plt.close(fig)
    return image_path


# Генерация изображений для каждого этапа для токена "black"
frame = 1  # Токен "black" находится на индексе 1
scalar_products = np.dot(query_vectors[frame], key_vectors.T)  # Скалярные произведения
softmax_weights = np.exp(scalar_products) / np.sum(np.exp(scalar_products))  # Softmax
weighted_values = np.dot(softmax_weights, value_vectors)  # Итоговый вектор

# Генерация изображений для каждого этапа
image_paths = []
for step in range(1, 5):
    image_paths.append(
        save_step_image(frame, step, query_vectors, key_vectors, value_vectors, softmax_weights, weighted_values))

image_paths
