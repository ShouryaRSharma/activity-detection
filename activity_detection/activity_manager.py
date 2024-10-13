import threading
from queue import Queue, Full, Empty
from activity_detection.inputs.camera_input import (
    CameraInputInterface,
    IPCamera,
    LocalCamera,
)
from activity_detection.processing.image_processing import (
    ImageProcessingInterface,
    DefaultImageProcessor,
)
from activity_detection.classifiers.activity_classifier import (
    ActivityDetectionInterface,
    MoondreamActivityDetector,
    YOLOActivityDetector,
)
from activity_detection.logging_config import setup_logger
from activity_detection.security.security_capture import DefaultVideoCapture
from activity_detection.security.security_logging import DefaultSecurityLogging
from activity_detection.security.security_module import SecurityModule


COMPONENT_MAPPING: dict[str, dict[str | None]] = {
    "camera_input": {
        "IPCamera": IPCamera,
        "LocalCamera": LocalCamera,
    },
    "image_processor": {
        "DefaultImageProcessor": DefaultImageProcessor,
    },
    "activity_detector": {
        "YOLOActivityDetector": YOLOActivityDetector,
        "MoondreamActivityDetector": MoondreamActivityDetector,
    },
    "security_logging": {
        "DefaultSecurityLogging": DefaultSecurityLogging,
    },
    "video_capture": {
        "DefaultVideoCapture": DefaultVideoCapture,
    },
}


class ActivityManager:
    def __init__(
        self,
        camera_input: CameraInputInterface,
        image_processor: ImageProcessingInterface,
        activity_detector: ActivityDetectionInterface,
        security_module: SecurityModule,
    ):
        self.camera_input = camera_input
        self.image_processor = image_processor
        self.activity_detector = activity_detector
        self.security_module = security_module
        self.frame_queue = Queue(maxsize=100)
        self.processed_frame_queue = Queue(maxsize=100)
        self.running = True
        self.logger = setup_logger(self.__class__.__name__)

    def capture_frames(self):
        self.camera_input.start_capture()
        while self.running:
            frame = self.camera_input.get_frame()
            try:
                self.frame_queue.put_nowait(frame)
            except Full:
                self.logger.warning("Frame queue is full. Skipping frame.")

    def process_frames(self):
        while self.running:
            try:
                frame = self.frame_queue.get()
                processed_image = self.image_processor.process_frame(frame)
                prediction = self.activity_detector.detect_activity(processed_image)
                try:
                    self.processed_frame_queue.put((frame, prediction))
                except Full:
                    self.logger.warning(
                        "Processed frame queue is full. Skipping frame."
                    )
            except Empty:
                pass

    def write_video(self):
        while self.running:
            try:
                frame, prediction = self.processed_frame_queue.get()
                self.security_module.process_frame(frame, prediction)
            except Empty:
                pass

    def run_activity_detection(self):
        try:
            capture_thread = threading.Thread(target=self.capture_frames)
            process_thread = threading.Thread(target=self.process_frames)
            write_thread = threading.Thread(target=self.write_video)

            capture_thread.start()
            process_thread.start()
            write_thread.start()

            capture_thread.join()
            process_thread.join()
            write_thread.join()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user. Stopping the program.")
        finally:
            self.running = False
            self.camera_input.stop_capture()
            self.security_module.video_capture.stop_video_capture()
            self.logger.info("Activity detection stopped.")

    @classmethod
    def from_config(cls, config: dict[str | None]):
        components = {}

        for component_type, component_config in config.items():
            if component_type == "security_module":
                continue

            class_name = next(iter(component_config))
            args = component_config[class_name].get("args", ())
            components[component_type] = cls.create_object(
                component_type, class_name, *args
            )

        security_module = SecurityModule(
            components["video_capture"], components["security_logging"]
        )

        return cls(
            components["camera_input"],
            components["image_processor"],
            components["activity_detector"],
            security_module,
        )

    @staticmethod
    def create_object(component_type: str, class_name: str, *args, **kwargs):
        if component_type not in COMPONENT_MAPPING:
            raise ValueError(f"Invalid component type: {component_type}")

        if class_name not in COMPONENT_MAPPING[component_type]:
            raise ValueError(f"Invalid class name for {component_type}: {class_name}")

        cls = COMPONENT_MAPPING[component_type][class_name]
        return cls(*args, **kwargs)
