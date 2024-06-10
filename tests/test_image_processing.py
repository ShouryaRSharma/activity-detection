import pytest
from camera_detection.image_processing import MoondreamImageProcessor


@pytest.mark.xfail(reason="Not implemented yet")
def test_process_frame():
    processor = MoondreamImageProcessor()
    frame = None  # Provide a sample frame for testing
    detection_results = processor.process_frame(frame)
    assert detection_results is not None
