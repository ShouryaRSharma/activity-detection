import datetime
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from activity_detection.logging_config import setup_logger


class VideoCaptureInterface(ABC):
    @abstractmethod
    def start_video_capture(self):
        pass

    @abstractmethod
    def stop_video_capture(self):
        pass

    @abstractmethod
    def capture_frame(self, frame: np.ndarray):
        pass


class DefaultVideoCapture(VideoCaptureInterface):
    def __init__(
        self,
        output_dir: str = "suspicious_activity_videos",
        max_clip_seconds: float = 300.0,
        fps: float = 30.0,
    ):
        if max_clip_seconds <= 0:
            raise ValueError("max_clip_seconds must be greater than zero")
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.max_clip_seconds = max_clip_seconds
        self.fps = fps
        self.video_writer = None
        self.output_file = None
        self.clip_started_at = None
        self.capturing = False
        self.logger = setup_logger(self.__class__.__name__)

    def start_video_capture(self):
        if self.capturing:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.capturing = True
        self.clip_started_at = None
        self.output_file = None

    def stop_video_capture(self):
        self._release_writer()
        self.capturing = False
        self.clip_started_at = None
        self.output_file = None

    def capture_frame(self, frame: np.ndarray):
        if not self.capturing:
            return
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be a BGR image array")

        try:
            if self.video_writer is None:
                self._open_writer(frame)
            elif (
                self.clip_started_at is not None
                and time.monotonic() - self.clip_started_at >= self.max_clip_seconds
            ):
                self._release_writer()
                self._open_writer(frame)

            self.video_writer.write(frame)
        except Exception:
            self._release_writer()
            raise

    def _open_writer(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.output_file = self.output_dir / f"suspicious_activity_{timestamp}.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            str(self.output_file), fourcc, self.fps, (width, height)
        )
        if not self.video_writer.isOpened():
            self.video_writer = None
            raise ValueError(f"Failed to open video writer for {self.output_file}")
        self.clip_started_at = time.monotonic()
        self.logger.info("Video capture started: %s", self.output_file)

    def _release_writer(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            if self.output_file is not None:
                self.logger.info("Video saved: %s", self.output_file)
