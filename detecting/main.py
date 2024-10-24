# https://rutube.ru/video/cbd82d6a3bf510fdc9b9a88aa740f9b3/
# https://colab.research.google.com/github/open-mmlab/mmaction2/blob/master/demo/mmaction2_tutorial.ipynb#scrollTo=ZPwKGzqydnb20
#  one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: gpu
import torch
import action_classifier
# import rutube_downloader
# from video_rescaller import change_video_resolution
# from dotenv import load_dotenv
import os

# load_dotenv()
# log_vk = os.getenv('LOGIN_VK')
# passwd_vk = os.getenv('PASSW_VK')
# print(log_vk)
# videoPath = rutube_downloader.download_rutube_video("https://rutube.ru/video/cbd82d6a3bf510fdc9b9a88aa740f9b3/")


input_video = './rutube_d/baza_video.mp4'
output_video = './rescaled/baza_video.mp4'
new_width = 224
new_height = 224

#change_video_resolution(input_video, output_video, new_width, new_height)
torch.backends.cuda.matmul.allow_tf32 = True
action_classifier.detect_aggressive_actions('./detecting/rutube_d/baza_video.mp4')
