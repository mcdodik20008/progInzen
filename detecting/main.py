# https://rutube.ru/video/cbd82d6a3bf510fdc9b9a88aa740f9b3/
import action_classifier
import rutube_downloader
from detecting.video_rescaller import change_video_resolution

# videoPath = rutube_downloader.download_rutube_video("https://rutube.ru/video/cbd82d6a3bf510fdc9b9a88aa740f9b3/")

# Параметры
input_video = './rutube_d/Подборка уличных драк и разборок!!! за весь 2021 драки.mp4'
output_video = './rescaled/Подборка уличных драк и разборок!!! за весь 2021 драки.mp4'
new_width = 224
new_height = 224

#change_video_resolution(input_video, output_video, new_width, new_height)

action_classifier.detect_aggressive_actions('./rescaled/Подборка уличных драк и разборок!!! за весь 2021 драки.mp4')

