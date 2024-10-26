from hugging_face_video_classifier import HuggingFaceVideoClassifier
from torch_vision_video_classifier import TorchVisionVideoClassifier


def get_classify_model(video_classifier_model, labels, device, fp16):
    if video_classifier_model in TorchVisionVideoClassifier.available_model_names():
        print("TORCH " + video_classifier_model)
        return TorchVisionVideoClassifier(video_classifier_model, device=device)
    else:
        print("HUGGING " + video_classifier_model)
        return HuggingFaceVideoClassifier(labels, model_name=video_classifier_model, device=device, fp16=fp16)