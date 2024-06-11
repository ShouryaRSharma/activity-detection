import threading
import time
from queue import Queue, Full, Empty
from typing import TypeVar
from activity_detection.camera_input import CameraInputInterface
from activity_detection.image_processing import ImageProcessingInterface
from activity_detection.activity_classifier import ActivityDetectionInterface
from activity_detection.logging_config import setup_logger
from activity_detection.security_module import SecurityModule

T = TypeVar("T")


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
        self.frame_queue = Queue(maxsize=100)  # Increased queue size
        self.processed_frame_queue = Queue(maxsize=100)  # Increased queue size
        self.running = True
        self.logger = setup_logger(self.__class__.__name__)

    def capture_frames(self):
        self.camera_input.start_capture()
        frame_count = 0
        frame_skipping_interval = 6  # Adjust this value as needed
        while self.running:
            frame = self.camera_input.get_frame()
            frame_count += 1
            if frame_count % frame_skipping_interval != 0:
                try:
                    self.frame_queue.put_nowait(frame)
                except Full:
                    self.logger.warning("Frame queue is full. Skipping frame.")
            time.sleep(0.1)  # Adding a delay to match the processing speed

    def process_frames(self):
        while self.running:
            try:
                frame = self.frame_queue.get(
                    timeout=1
                )  # Wait for a frame if the queue is empty
                processed_image = self.image_processor.process_frame(frame)
                activity_detected = self.activity_detector.detect_activity(
                    processed_image
                )
                try:
                    self.processed_frame_queue.put_nowait((frame, activity_detected))
                except Full:
                    self.logger.warning(
                        "Processed frame queue is full. Skipping frame."
                    )
            except Empty:
                pass

    def write_video(self):
        while self.running:
            try:
                frame, activity_detected = self.processed_frame_queue.get(
                    timeout=1
                )  # Wait for a frame if the queue is empty
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