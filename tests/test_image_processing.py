import numpy as np
from PIL import Image
from activity_detection.processing.image_processing import DefaultImageProcessor


def test_process_frame():
    mock_frame = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8
    )
    image_processor = DefaultImageProcessor()

    processed_image = image_processor.process_frame(mock_frame)

    assert isinstance(processed_image, Image.Image)
    assert processed_image.size == (1920, 1080)


def test_preprocess_frame():
    mock_frame = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8
    )
    image_processor = DefaultImageProcessor()

    preprocessed_image = image_processor._preprocess_frame(mock_frame)

    assert isinstance(preprocessed_image, Image.Image)
    assert preprocessed_image.size == (1920, 1080)
