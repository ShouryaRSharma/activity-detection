from abc import ABC, abstractmethod


class VideoCaptureInterface(ABC):
    @abstractmethod
    def start_video_capture(self):
        pass

    @abstractmethod
    def stop_video_capture(self):
        pass

    @abstractmethod
    def save_video(self, video_data):
        pass


class DefaultVideoCapture(VideoCaptureInterface):
    def start_video_capture(self):
        # TODO: Implement starting video capture
        pass

    def stop_video_capture(self):
        # TODO: Implement stopping video capture
        pass

    def save_video(self, video_data):
        # TODO: Implement saving the captured video data
        pass


class SecurityLogsInterface(ABC):
    @abstractmethod
    def log_suspicious_activity(self, detection_results):
        pass
