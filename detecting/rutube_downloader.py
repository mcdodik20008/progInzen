import youtube_dl
import os

def download_rutube_video(video_url, save_directory = './rutube_d'):
    # Создаем директорию, если ее нет
    os.makedirs(save_directory, exist_ok=True)
    outtmpl = os.path.join(save_directory, '%(title)s.%(ext)s')
    # Опции для скачивания
    ydl_opts = {
        'format': 'best',
        'outtmpl': outtmpl,  # Шаблон имени файла с оригинальным названием
    }
    print("Начал скачку!")
    # Скачиваем видео
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return outtmpl
    print("Видео с Рутуба скачано успешно!")
