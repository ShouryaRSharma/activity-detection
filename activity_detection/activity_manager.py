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
                activity_detected = self.activity_detector.detect_activity(
                    processed_image
                )
                try:
                    self.processed_frame_queue.put((frame, activity_detected))
                except Full:
                    self.logger.warning(
                        "Processed frame queue is full. Skipping frame."
                    )
            except Empty:
                pass

    def write_video(self):
        while self.running:
            try:
                frame, activity_detected = self.processed_frame_queue.get()
                self.security_module.process_frame(frame, activity_detected)
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
        camera_input_config = config["camera_input"]
        camera_input_class = next(iter(camera_input_config))
        camera_input_args = camera_input_config[camera_input_class].get("args", ())
        camera_input = cls.create_object(
            "camera_input", camera_input_class, *camera_input_args
        )

        image_processor_config = config["image_processor"]
        image_processor_class = next(iter(image_processor_config))
        image_processor = cls.create_object("image_processor", image_processor_class)

        activity_detector_config = config["activity_detector"]
        activity_detector_class = next(iter(activity_detector_config))
        activity_detector_args = activity_detector_config[activity_detector_class].get(
            "args", ()
        )
        activity_detector = cls.create_object(
            "activity_detector", activity_detector_class, *activity_detector_args
        )

        security_logging_config = config["security_logging"]
        security_logging_class = next(iter(security_logging_config))
        security_logging = cls.create_object("security_logging", security_logging_class)

        video_capture_config = config["video_capture"]
        video_capture_class = next(iter(video_capture_config))
        video_capture_args = video_capture_config[video_capture_class].get("args", ())
        video_capture = cls.create_object(
            "video_capture", video_capture_class, *video_capture_args
        )

        security_module = SecurityModule(video_capture, security_logging)

        return cls(camera_input, image_processor, activity_detector, security_module)

    @staticmethod
    def create_object(component_type: str, class_name: str, *args, **kwargs):
        if component_type not in COMPONENT_MAPPING:
            raise ValueError(f"Invalid component type: {component_type}")

        if class_name not in COMPONENT_MAPPING[component_type]:
            raise ValueError(f"Invalid class name for {component_type}: {class_name}")

        cls = COMPONENT_MAPPING[component_type][class_name]
        return cls(*args, **kwargs)
