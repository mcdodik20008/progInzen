from controldetecting.model.ColorNameToCode import ColorNameToCode


class AnnotateData:

    def __init__(self, labels, confs, label_to_cname):
        self.labels = labels
        self.confs = confs
        self.label_to_cname = label_to_cname

    def __iter__(self):
        for label, conf in zip(self.labels, self.confs):
            top_preds = sorted(zip(label, conf), key=lambda x: x[1], reverse=True)
            label_text = "|".join(f"{label} ({conf:.2f})" for label, conf in top_preds if conf > 0.19)
            val = top_preds[0][0]
            yield label_text, ColorNameToCode.get_color_by_name(self.label_to_cname[val])

    @staticmethod
    def get_template(count, label, color_name):
        labels = [[label] for _ in range(count)]
        confs = [[1.] for _ in range(count)]
        label_to_cname = {label: color_name}
        return AnnotateData(labels, confs, label_to_cname)
