import torch
from torchvision import transforms
from ultralytics import YOLO  # Предполагается, что используется библиотека YOLOv5 или YOLOv8 с PyTorch


class PersonDetectorYOLOv11:
    def __init__(self, model_name, device):
        self.device = device

        self.model = YOLO(model_name)
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

    def __call__(self, frame):
        """
        Метод для детекции людей в кадре.
        """
        with torch.no_grad():
            outputs = self.model.track(frame, persist=True, classes=[0])

        if len(outputs) > 0 and outputs[0].boxes.id is not None:
            boxes = outputs[0].boxes.xyxy.cpu().numpy()
            track_ids = outputs[0].boxes.id.cpu().numpy()
            classes = outputs[0].boxes.cls.cpu().numpy()
            return boxes, track_ids, classes
        return None, None, None
