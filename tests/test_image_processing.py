from unittest.mock import patch
import numpy as np
from PIL import Image
from activity_detection.image_processing import MoondreamImageProcessor


@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_process_frame(_, mock_model):
    mock_frame = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8
    )
    mock_enc_image = np.array([1, 2, 3])
    mock_model.return_value.encode_image.return_value = mock_enc_image
    image_processor = MoondreamImageProcessor("model_id", "revision")

    enc_image = image_processor.process_frame(mock_frame)

    mock_model.return_value.encode_image.assert_called_once()
    assert np.array_equal(enc_image, mock_enc_image)


@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_preprocess_frame(_, __):
    mock_frame = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8
    )
    image_processor = MoondreamImageProcessor("model_id", "revision")
    preprocessed_image = image_processor._preprocess_frame(mock_frame)
    assert isinstance(preprocessed_image, Image.Image)
    assert preprocessed_image.size == (2, 2)
