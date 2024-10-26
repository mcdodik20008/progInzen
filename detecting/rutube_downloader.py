import youtube_dl
import os

def download_rutube_video(video_url, save_path ='detecting/rutube_d'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    out_tmpl = os.path.join(save_path)
    if os.path.exists(out_tmpl):
        raise FileExistsError("Такой файл уже есть, пж не затирай его")

    ydl_opts = {
        'format': 'best',
        'outtmpl': out_tmpl,
    }

    print("Начал скачку!")
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return out_tmpl
    print("Видео с Рутуба скачано успешно!")
