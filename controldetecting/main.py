from ultralytics.utils.torch_utils import select_device

from BehaviorClassifier import BehaviorClassifier
from CapProcessor import CapProcessor
from PersonDetectorYOLOv11 import PersonDetectorYOLOv11
from VideoBehaviorAnalyzer import VideoAnalyzer

input_video = "rutube_d/baza3.mp4"
output_video = "results/output_video_with_behavior.mp4"

detection_model_name = "microsoft/xclip-base-patch32"
device = select_device("cuda")

detector = PersonDetectorYOLOv11("yolo11x.pt", device)
cap_processor = CapProcessor(detection_model_name, device)
classifier = BehaviorClassifier(cap_processor, device, detection_model_name)
analyzer = VideoAnalyzer(detector, classifier, cap_processor)

analyzer.analyze_video(input_video, output_video, False)
