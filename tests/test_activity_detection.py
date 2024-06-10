import pytest
from camera_detection.activity_detection import SuspiciousActivityDetector


@pytest.mark.xfail(reason="Not implemented yet")
def test_detect_suspicious_activity():
    detector = SuspiciousActivityDetector()
    detection_results = None  # Provide sample detection results for testing
    suspicious_activity = detector.detect_activity(detection_results)
    assert suspicious_activity is not None
