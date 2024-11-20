from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

from BehaviorClassifier import BehaviorClassifier
from CapProcessor import CapProcessor
from PersonDetectorYOLOv11 import PersonDetectorYOLOv11
from VideoBehaviorAnalyzer import VideoAnalyzer
from controldetecting.CarAccidentClassifier import CarAccidentClassifier

# in_dir = "rutube_d/"
in_dir = "onlinecams/"

input_video = in_dir + "baza4.mp4"
output_video = "results/output_video_with_behavior.mp4"

# peter881122/xclip-base-patch32-finetuned-custom-subset
# asmaa1/xclip-base-patch32-finetuned-SLT-subset
# microsoft/xclip-base-patch32
# microsoft/xclip-base-patch16
processor_name = "microsoft/xclip-base-patch16-kinetics-600"
detection_model_name = "microsoft/xclip-base-patch16-kinetics-600"
device = select_device("cuda")

detector = PersonDetectorYOLOv11("yolo11x.pt", device)
cap_processor = CapProcessor(processor_name, device)
classifier = BehaviorClassifier(cap_processor, device, detection_model_name, processor_name)
car_accident = CarAccidentClassifier()
analyzer = VideoAnalyzer(detector, classifier, car_accident, cap_processor)

analyzer.analyze_video(input_video, output_video, False)

# https://microsoft.github.io/AirSim/build_windows/