from abc import abstractmethod, ABC


class Classifier(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def try_reset_to_predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_data_for_annotate(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_class_id(self, *args, **kwargs):
        pass
