from ultralytics.utils.torch_utils import select_device

from controldetecting.BehaviorClassifier import BehaviorClassifier
from controldetecting.CapProcessor import CapProcessor
from controldetecting.PersonDetectorYOLOv11 import PersonDetectorYOLOv11
from controldetecting.VideoBehaviorAnalyzer import VideoAnalyzer

detection_model_name = "microsoft/xclip-base-patch32"
device = select_device("cuda")

detector = PersonDetectorYOLOv11("yolo11x.pt", device)
cap_processor = CapProcessor(detection_model_name, device)
classifier = BehaviorClassifier(cap_processor, device, detection_model_name)
analyzer = VideoAnalyzer(detector, classifier, cap_processor)

analyzer.analyze_video("detecting/rescaled/baza3.mp4", "results/output_video_with_behavior.mp4", False)
