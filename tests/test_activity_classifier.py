from unittest.mock import patch, MagicMock
from PIL import Image
import pytest
from activity_detection.classifiers.activity_classifier import (
    MoondreamActivityDetector,
    YOLOActivityDetector,
)


@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_detect_activity_with_person_close(mock_tokenizer, mock_model):
    mock_model.return_value.to.return_value.answer_question.return_value = "YES"
    detector = MoondreamActivityDetector("model_id", "revision")
    image = Image.new("RGB", (100, 100))

    # Act
    result = detector.detect_activity(image)

    # Assert
    assert result is True
    mock_model.return_value.to.return_value.answer_question.assert_called_once_with(
        mock_model.return_value.to.return_value.encode_image.return_value,
        "Is there a person close to the camera in this image? (Only answer 'YES' or 'NO')",
        mock_tokenizer.return_value,
    )


@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_detect_activity_without_person_close(mock_tokenizer, mock_model):
    # Arrange
    mock_model.return_value.to.return_value.answer_question.return_value = "NO"
    detector = MoondreamActivityDetector("model_id", "revision")
    image = Image.new("RGB", (100, 100))

    # Act
    result = detector.detect_activity(image)

    # Assert
    assert result is False
    mock_model.return_value.to.return_value.answer_question.assert_called_once_with(
        mock_model.return_value.to.return_value.encode_image.return_value,
        "Is there a person close to the camera in this image? (Only answer 'YES' or 'NO')",
        mock_tokenizer.return_value,
    )


@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_detect_activity_with_invalid_answer(mock_tokenizer, mock_model):
    # Arrange
    mock_model.return_value.answer_question.return_value = "INVALID"
    detector = MoondreamActivityDetector("model_id", "revision")
    image = Image.new("RGB", (100, 100))

    with pytest.raises(ValueError, match="Invalid answer from the model"):
        detector.detect_activity(image)


@patch("activity_detection.classifiers.activity_classifier.YOLOWorld")
def test_detect_activity_with_person_detected(mock_yolo):
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.conf = 0.8
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "person"}
    mock_model.predict.return_value = [mock_result]
    mock_yolo.return_value.to.return_value = mock_model
    detector = YOLOActivityDetector()
    image = Image.new("RGB", (100, 100))

    # Act
    result = detector.detect_activity(image)
    # Assert
    assert result is True
    mock_model.predict.assert_called_once_with(image)


@patch("activity_detection.classifiers.activity_classifier.YOLOWorld")
def test_detect_activity_without_person_detected(mock_yolo):
    # Arrange
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.conf = 0.8
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "car"}
    mock_model.predict.return_value = [mock_result]
    mock_yolo.return_value.to.return_value = mock_model
    detector = YOLOActivityDetector()
    image = Image.new("RGB", (100, 100))

    # Act
    result = detector.detect_activity(image)

    # Assert
    assert result is False
    mock_model.predict.assert_called_once_with(image)


@patch("activity_detection.classifiers.activity_classifier.YOLOWorld")
def test_detect_activity_with_low_confidence(mock_yolo):
    # Arrange
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.conf = 0.3
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "person"}
    mock_model.predict.return_value = [mock_result]
    mock_yolo.return_value.to.return_value = mock_model
    detector = YOLOActivityDetector()
    image = Image.new("RGB", (100, 100))

    # Act
    result = detector.detect_activity(image)

    # Assert
    assert result is False
    mock_model.predict.assert_called_once_with(image)
