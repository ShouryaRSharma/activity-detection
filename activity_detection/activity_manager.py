import threading
from queue import Empty, Full, Queue
from typing import Any

from activity_detection.classifiers import (
    MoondreamActivityDetector,
    YOLOActivityDetector,
    YOLOWorldActivityDetector,
)
from activity_detection.classifiers.interfaces import (
    ActivityDetectionInterface,
)
from activity_detection.inputs.camera_input import (
    CameraInputInterface,
    IPCamera,
    LocalCamera,
)
from activity_detection.logging_config import setup_logger
from activity_detection.processing.image_processing import (
    DefaultImageProcessor,
    ImageProcessingInterface,
)
from activity_detection.security.security_capture import DefaultVideoCapture
from activity_detection.security.security_logging import DefaultSecurityLogging
from activity_detection.security.security_module import SecurityModule

COMPONENT_MAPPING: dict[str, dict[str | None, type]] = {
    "camera_input": {
        "IPCamera": IPCamera,
        "LocalCamera": LocalCamera,
    },
    "image_processor": {
        "DefaultImageProcessor": DefaultImageProcessor,
    },
    "activity_detector": {
        "YOLOActivityDetector": YOLOActivityDetector,
        "YOLOWorldActivityDetector": YOLOWorldActivityDetector,
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
        self.stop_event = threading.Event()
        self._thread_errors: list[BaseException] = []
        self._error_lock = threading.Lock()
        self.logger = setup_logger(self.__class__.__name__)

    def _record_error(self, exc: BaseException):
        with self._error_lock:
            self._thread_errors.append(exc)
        self.logger.exception("Worker thread failed: %s", exc)
        self.stop_event.set()

    def capture_frames(self):
        try:
            self.camera_input.start_capture()
            while not self.stop_event.is_set():
                frame = self.camera_input.get_frame()
                try:
                    self.frame_queue.put(frame, timeout=0.5)
                except Full:
                    self.logger.warning("Frame queue is full. Skipping frame.")
        except Exception as exc:  # noqa: BLE001
            self._record_error(exc)

    def process_frames(self):
        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                except Empty:
                    continue
                processed_image = self.image_processor.process_frame(frame)
                prediction = self.activity_detector.detect_activity(processed_image)
                try:
                    self.processed_frame_queue.put((frame, prediction), timeout=0.5)
                except Full:
                    self.logger.warning(
                        "Processed frame queue is full. Skipping frame."
                    )
        except Exception as exc:  # noqa: BLE001
            self._record_error(exc)

    def write_video(self):
        try:
            while not self.stop_event.is_set():
                try:
                    frame, prediction = self.processed_frame_queue.get(timeout=0.5)
                except Empty:
                    continue
                self.security_module.process_frame(frame, prediction)
        except Exception as exc:  # noqa: BLE001
            self._record_error(exc)

    def run_activity_detection(self):
        threads = [
            threading.Thread(target=self.capture_frames, daemon=True),
            threading.Thread(target=self.process_frames, daemon=True),
            threading.Thread(target=self.write_video, daemon=True),
        ]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user. Stopping the program.")
        finally:
            self.stop_event.set()
            self.camera_input.stop_capture()
            self.security_module.video_capture.stop_video_capture()
            for thread in threads:
                if thread.is_alive():
                    thread.join(timeout=5)
            self.logger.info("Activity detection stopped.")

        if self._thread_errors:
            raise RuntimeError("One or more worker threads failed") from (
                self._thread_errors[0]
            )

    @classmethod
    def from_config(cls, config: dict[str, Any]):
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
