import cv2


def change_video_resolution(input_video_path, output_video_path, width, height):
    # Открываем входное видео
    cap = cv2.VideoCapture(input_video_path)

    # Получаем параметры входного видео
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Кодек для сохранения видео
    fps = cap.get(cv2.CAP_PROP_FPS)  # Частота кадров
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Изменяем размер кадра
        resized_frame = cv2.resize(frame, (width, height))

        # Записываем измененный кадр в выходное видео
        out.write(resized_frame)

    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()
