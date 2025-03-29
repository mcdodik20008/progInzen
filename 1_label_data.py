from inference.behavior_classifier import BehaviorClassifier
from tools import label_videos as lv

classifier = BehaviorClassifier()
lv.label_frame_folders(classifier)
