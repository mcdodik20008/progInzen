class ColorNameToCode:
    cname_to_color = {
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "blue": (255, 0, 0),
    }

    @staticmethod
    def get_color_by_name(name):
        return ColorNameToCode.cname_to_color[name]
