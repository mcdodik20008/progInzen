from ultralytics.utils.plotting import Annotator


class FrameAnnotator:
    def __init__(self):
        self.annotator = None

    def __call__(self, frame, zipped_data, label_to_colors):
        self.annotator = Annotator(frame, line_width=3, font_size=10, pil=False)
        for box, pred_label, pred_conf, class_num in zipped_data:
            self.annotate_frame_one_box(box, pred_label, pred_conf, class_num, label_to_colors)

    def annotate_frame_one_box(self, box, pred_label, pred_conf, class_num, label_to_colors):
        top2_preds = sorted(zip(pred_label, pred_conf), key=lambda x: x[1], reverse=True)
        label_text = " | ".join([f"{label} ({conf:.2f})" if conf > 0.3 else "" for label, conf in top2_preds])
        label, conf = top2_preds[0]
        label_text = label_text + f"class: {class_num}"
        self.annotator.box_label(box, label_text, color=label_to_colors[label], font_size=10)