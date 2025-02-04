import numpy as np
import matplotlib.pyplot as plt

# Входные токены и их вектора (предположим, что мы используем 2D-вектора)
tokens = ["A", "black", "cat"]
input_vectors = np.array([[0.2, 0.5], [0.1, 0.6], [0.4, -0.2]])


# Простейший механизм Self-Attention (по аналогии с тем, что было раньше)
def self_attention_step(input_vectors):
    # Для упрощения используем фиксированные запросы, ключи и значения (Q, K, V)
    query_vectors = input_vectors  # Запросы
    key_vectors = input_vectors  # Ключи
    value_vectors = input_vectors  # Значения

    # Вычисляем внимание как скалярные произведения
    attention_weights = np.dot(query_vectors, key_vectors.T)  # Скалярные произведения
    attention_weights = np.exp(attention_weights)  # Применяем exp для числовой стабильности
    attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)  # Softmax

    # Применяем веса внимания к значениям
    output_vectors = np.dot(attention_weights, value_vectors)
    return output_vectors


# Генерация изображений для каждого слоя
def plot_attention_layers(input_vectors, num_layers=3):
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)

    # Отображаем токены и их вектора на первом слое
    ax.scatter(input_vectors[:, 0], input_vectors[:, 1], color='blue', label="Input Tokens", s=100)
    for i, token in enumerate(tokens):
        ax.text(input_vectors[i, 0], input_vectors[i, 1], f'{token}', fontsize=12, ha='center')

    ax.set_title("Многоуровневая обработка токенов через Self-Attention", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Отображаем результат обработки после каждого слоя
    current_vectors = input_vectors
    for layer in range(1, num_layers + 1):
        current_vectors = self_attention_step(current_vectors)
        ax.scatter(current_vectors[:, 0], current_vectors[:, 1], label=f"Layer {layer} Output", s=100)
        for i, token in enumerate(tokens):
            ax.text(current_vectors[i, 0], current_vectors[i, 1], f'{token}', fontsize=12, ha='center')

    ax.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    plt.show()


# Визуализируем 3 слоя
plot_attention_layers(input_vectors, num_layers=3)
