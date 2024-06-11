from activity_detection.inputs.camera_input import IPCamera
from activity_detection.processing.image_processing import MoondreamImageProcessor
from activity_detection.classifiers.activity_classifier import MoondreamActivityDetector
from activity_detection.security.security_module import (
    SecurityModule,
    DefaultVideoCapture,
    DefaultSecurityLogging,
)
from activity_detection.activity_manager import ActivityManager


def main():
    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"
    ipv4_address = "http://192.168.1.111:8080/video"
    camera_input = IPCamera(ipv4_address)
    image_processor = MoondreamImageProcessor()
    activity_detector = MoondreamActivityDetector(model_id, revision)
    security_logging = DefaultSecurityLogging()
    video_capture = DefaultVideoCapture("suspicious_activity_videos")
    security_module = SecurityModule(video_capture, security_logging)

    activity_manager = ActivityManager(
        camera_input, image_processor, activity_detector, security_module
    )
    activity_manager.run_activity_detection()


if __name__ == "__main__":
    main()
