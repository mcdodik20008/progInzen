from ultralytics.utils.plotting import Annotator

from controldetecting.model.AnnotateData import AnnotateData


class FrameAnnotator:
    def __init__(self):
        self.annotator = None

    def __call__(self, frame, boxes, data: AnnotateData):
        self.annotator = Annotator(frame, line_width=2, font_size=10, pil=False)

        n = 0
        len_boxes = len(boxes)
        for (box, (text, color)) in zip(boxes, data):
            n += 1
            if box[0] == 0 and box[1] == 0:
                print(f"ОБЩАЯ ОЦЕНКА ОБСТАНОВКИ НА ВИДЕО: {text}, размер кадра: {box}")
            self.annotator.box_label(box, text, color=color)
