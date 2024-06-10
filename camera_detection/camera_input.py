from abc import ABC, abstractmethod


class CameraInputInterface(ABC):
    @abstractmethod
    def start_capture(self):
        pass

    @abstractmethod
    def stop_capture(self):
        pass

    @abstractmethod
    def get_frame(self):
        pass


class IPCamera(CameraInputInterface):
    def start_capture(self):
        # TODO: Implement starting the live stream capture from the IP camera
        pass

    def stop_capture(self):
        # TODO: Implement stopping the live stream capture
        pass

    def get_frame(self):
        # TODO: Implement retrieving the current frame from the live stream
        pass
