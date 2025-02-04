# Матрица внимания S
import numpy as np
from prettytable import PrettyTable

S = np.array([[10.78, 14.08],
              [13.72, 17.92]])

# Функция Softmax по строкам
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Стабильный softmax
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# Применяем Softmax к каждой строке матрицы S
alpha = softmax(S)  # Нормализованная матрица внимания

table = PrettyTable()
for row in alpha:
    table.add_row(row)

print(f"alpha: {table}")
