from ultralytics.utils.torch_utils import select_device

from CapProcessor import CapProcessor
from PersonDetectorYOLOv11 import PersonDetectorYOLOv11
from controldetecting.CarAccidentClassifier import CarAccidentClassifier
from controldetecting.FrameAnnotator import FrameAnnotator
from controldetecting.VideoAnalyzer import VideoAnalyzer
from controldetecting.analyzer.XclipClassifier import XclipClassifier

# in_dir = "rutube_d/"
in_dir = "onlinecams/"

input_video = in_dir + "baza4.mp4"
# input_video = 0

output_video = "results/output_video_with_behavior.mp4"
# peter881122/xclip-base-patch32-finetuned-custom-subset
# asmaa1/xclip-base-patch32-finetuned-SLT-subset
# microsoft/xclip-base-patch32
# microsoft/xclip-base-patch16
# microsoft/xclip-base-patch16-kinetics-600
# microsoft/xclip-large-patch14-kinetics-600
processor_name = "microsoft/xclip-large-patch14-kinetics-600"
detection_model_name = "microsoft/xclip-large-patch14-kinetics-600"
device = select_device("cuda")

detector = PersonDetectorYOLOv11("yolo11x.pt", device)
cap_processor = CapProcessor(processor_name, device)
frame_annotator = FrameAnnotator()
classifier = XclipClassifier(cap_processor, frame_annotator, device, detection_model_name, processor_name)
car_accident = CarAccidentClassifier()
analyzer = VideoAnalyzer(detector, cap_processor, frame_annotator, classifier, car_accident)

analyzer.analyze_video(input_video, output_video, False)

# https://microsoft.github.io/AirSim/build_windows/
