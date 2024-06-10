from abc import ABC, abstractmethod


class ImageProcessingInterface(ABC):
    @abstractmethod
    def process_frame(self, frame):
        pass


class MoondreamImageProcessor(ImageProcessingInterface):
    def process_frame(self, frame):
        # TODO: Implement processing the frame using the Moondream model
        pass
