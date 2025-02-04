from abc import abstractmethod, ABC


class Classifier(ABC):

    @abstractmethod
    def analyze_video(self, video_path, output_path, save_video=False):
        pass