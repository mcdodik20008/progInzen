import vk_api
import requests
import os

def download_vk_video(login, password, video_id, save_directory = './vk_d'):
    # Авторизация
    vk_session = vk_api.VkApi(login, password)
    vk = vk_session.get_api()

    # Получаем информацию о видео
    video_info = vk.video.get(video_ids=video_id)

    # Получаем URL видео и оригинальное название
    video_url = video_info['items'][0]['player']
    video_title = video_info['items'][0]['title']

    # Создаем директорию, если ее нет
    os.makedirs(save_directory, exist_ok=True)
    outtmpl = os.path.join(save_directory, f'{video_title}.mp4')

    # Скачиваем видео
    response = requests.get(video_url)
    with open(outtmpl, 'wb') as file:
        file.write(response.content)

    print("Видео с ВКонтакте скачано успешно!")
    return outtmpl
