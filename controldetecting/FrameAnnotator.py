from ultralytics.utils.plotting import Annotator


class FrameAnnotator:
    def __init__(self):
        self.annotator = None

    def annotate_frame(self, frame, zipped_data, label_to_colors):
        self.annotator = Annotator(frame, line_width=2, font_size=10, pil=False)
        for box, pred_label, pred_conf, class_num in zipped_data:
            self.annotate_frame_one_box(box, pred_label, pred_conf, class_num, label_to_colors)

    def annotate_frame_one_box(self, box, pred_label, pred_conf, class_num, label_to_colors):
        top2_preds = sorted(zip(pred_label, pred_conf), key=lambda x: x[1], reverse=True)
        label_text = "|".join([f"{label} ({conf:.2f})" if conf > 0.3 else "" for label, conf in top2_preds])
        label, conf = top2_preds[0]
        self.annotator.box_label(box, label_text, color=label_to_colors[label])

    def annotate_frame_one_box_by_text(self, boxes, text, color=(255, 0, 0)):
        for box in boxes:
            self.annotator.box_label(box, text, color=color)

    def __call__(self, box, text, color):
        pass