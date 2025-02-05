import os

import requests
import vk_api


def download_vk_video(login, password, video_id, save_directory='./vk_d'):
    vk_session = vk_api.VkApi(login, password)
    vk = vk_session.get_api()

    video_info = vk.video.get(video_ids=video_id)

    video_url = video_info['items'][0]['player']
    video_title = video_info['items'][0]['title']

    os.makedirs(save_directory, exist_ok=True)
    outtmpl = os.path.join(save_directory, f'{video_title}.mp4')

    response = requests.get(video_url)
    with open(outtmpl, 'wb') as file:
        file.write(response.content)

    print("Видео с ВКонтакте скачано успешно!")
    return outtmpl
