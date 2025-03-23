import os
import youtube_dl

def download_youtube_video(video_url, save_path='./video/video.mp4'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ydl_opts = {
        'format': 'best',
        'outtmpl': save_path,
    }

    print("Начал скачку!")
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    print("Видео скачано успешно!")
    return save_path


download_youtube_video("",'./video/video.mp4')