import os

from mmaction.apis import init_recognizer, inference_recognizer


def detect_aggressive_actions(video_path, result_path = './results'):
    # Параметры конфигурации модели
    config_file = './config/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
    checkpoint_file = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'
    # Инициализация модели
    model = load_model(config_file, checkpoint_file)
    print("Инициализировал модель")

    # Классификация действий
    results = classify_actions(model, video_path)
    print("Закончил классификацию")

    # Фильтрация агрессивных действий и запись результатов
    aggressive_actions = filter_aggressive_actions(results)
    print("Закончил фильтрацию")

    if result_path == "console":
        print_results(aggressive_actions)
        return

    # Сохранение результатов в файл
    fileName = os.path.splitext(os.path.basename(video_path))[0]
    with open(result_path + os.sep + fileName + '.txt', 'w') as f:
        for timestamp, label in aggressive_actions:
            f.write(f"{timestamp}: {label}\n")


def load_model(config_file, checkpoint_file, device='cuda:0'):
    """ Инициализирует модель для распознавания действий. """
    model = init_recognizer(config_file, checkpoint_file, device=device)
    return model


def classify_actions(model, video_path):
    """ Классифицирует действия в видео с помощью модели. """
    results = inference_recognizer(model, video_path)
    return results


def filter_aggressive_actions(results):
    """
    Фильтрует агрессивные действия из результатов классификации.

    :param results: Результаты классификации.
    :return: Список агрессивных действий с временными метками.
    """
    aggressive_labels = ["Aggressive", "Fight", "Attack"]  # Укажите ваши агрессивные действия
    aggressive_actions = []

    for i, result in enumerate(results):
        label = result['label']  # Предполагается, что результат содержит ключ 'label'
        if label in aggressive_labels:
            timestamp = i * 0.5  # Примерное вычисление временной метки (если кадры каждые 0.5 секунды)
            aggressive_actions.append((timestamp, label))

    return aggressive_actions


def print_results(results):
    """ Печатает результаты агрессивных действий. """
    print("Агрессивные действия (с временными метками):")
    for timestamp, label in results:
        print(f"{timestamp:.2f} секунд: {label}")
