from activity_detection.inputs.camera_input import IPCamera
from activity_detection.processing.image_processing import DefaultImageProcessor
from activity_detection.classifiers.activity_classifier import YOLOActivityDetector
from activity_detection.security.security_capture import DefaultVideoCapture
from activity_detection.security.security_logging import DefaultSecurityLogging
from activity_detection.security.security_module import SecurityModule
from activity_detection.activity_manager import ActivityManager


def main():
    ipv4_address = "http://192.168.1.111:8080/video"
    camera_input = IPCamera(ipv4_address)
    image_processor = DefaultImageProcessor()
    activity_detector = YOLOActivityDetector()
    security_logging = DefaultSecurityLogging()
    video_capture = DefaultVideoCapture("suspicious_activity_videos")
    security_module = SecurityModule(video_capture, security_logging)

    activity_manager = ActivityManager(
        camera_input, image_processor, activity_detector, security_module
    )
    activity_manager.run_activity_detection()


if __name__ == "__main__":
    main()
