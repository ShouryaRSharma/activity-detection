from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from activity_detection.classifiers import (
    MoondreamActivityDetector,
    YOLOActivityDetector,
)
from activity_detection.classifiers.types import Prediction
from activity_detection.classifiers.yolo_classifiers import (
    validate_model_reference,
)

PINNED_REVISION = "a" * 40


@pytest.fixture
def mock_image():
    return Image.new("RGB", (100, 100))


@pytest.fixture
def mock_moondream_model():
    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model,
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
    ):
        mock_model.return_value.to.return_value.answer_question.return_value = "YES"
        yield mock_model, mock_tokenizer


@pytest.fixture
def mock_yolo_model():
    with patch(
        "activity_detection.classifiers.yolo_classifiers.YOLOActivityDetector.load_model"
    ) as mock_yolo:
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.conf = 0.8
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "person"}
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value.to.return_value = mock_model
        yield mock_yolo


@pytest.mark.parametrize(
    "detector_class, mock_model_fixture, expected_detected",
    [
        (MoondreamActivityDetector, "mock_moondream_model", True),
        (YOLOActivityDetector, "mock_yolo_model", True),
    ],
)
def test_detect_activity_with_person(
    detector_class, mock_model_fixture, expected_detected, mock_image, request
):
    mock_model = request.getfixturevalue(mock_model_fixture)

    if detector_class == MoondreamActivityDetector:
        detector = detector_class(
            "vikhyatk/moondream2", PINNED_REVISION, allow_remote_code=True
        )
    else:
        detector = detector_class()

    result = detector.detect_activity(mock_image)

    assert isinstance(result, Prediction)
    assert result.detected is expected_detected

    if detector_class == MoondreamActivityDetector:
        mock_model[0].return_value.to.return_value.answer_question.assert_called_once()
    else:
        mock_model.return_value.to.return_value.predict.assert_called_once_with(
            mock_image
        )


@pytest.mark.parametrize(
    "detector_class, mock_model_fixture, model_output, expected_detected",
    [
        (MoondreamActivityDetector, "mock_moondream_model", "NO", False),
        (YOLOActivityDetector, "mock_yolo_model", {0: "car"}, False),
    ],
)
def test_detect_activity_without_person(
    detector_class,
    mock_model_fixture,
    model_output,
    expected_detected,
    mock_image,
    request,
):
    mock_model = request.getfixturevalue(mock_model_fixture)

    if detector_class == MoondreamActivityDetector:
        mock_model[
            0
        ].return_value.to.return_value.answer_question.return_value = model_output
        detector = detector_class(
            "vikhyatk/moondream2", PINNED_REVISION, allow_remote_code=True
        )
    else:
        mock_model.return_value.to.return_value.predict.return_value[
            0
        ].names = model_output
        detector = detector_class()

    result = detector.detect_activity(mock_image)

    assert isinstance(result, Prediction)
    assert result.detected is expected_detected


@pytest.mark.parametrize(
    "detector_class, mock_model_fixture, model_output",
    [
        (MoondreamActivityDetector, "mock_moondream_model", "INVALID"),
        (YOLOActivityDetector, "mock_yolo_model", 0.3),
    ],
)
def test_detect_activity_edge_cases(
    detector_class, mock_model_fixture, model_output, mock_image, request
):
    mock_model = request.getfixturevalue(mock_model_fixture)

    if detector_class == MoondreamActivityDetector:
        mock_model[
            0
        ].return_value.to.return_value.answer_question.return_value = model_output
        detector = detector_class(
            "vikhyatk/moondream2", PINNED_REVISION, allow_remote_code=True
        )
        with pytest.raises(ValueError, match="Invalid answer from the model"):
            detector.detect_activity(mock_image)
    else:
        mock_model.return_value.to.return_value.predict.return_value[0].boxes[
            0
        ].conf = model_output
        detector = detector_class()
        result = detector.detect_activity(mock_image)
        assert result.detected is False


def test_moondream_disables_remote_code_by_default():
    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model,
        patch("transformers.AutoTokenizer.from_pretrained"),
    ):
        mock_model.return_value.to.return_value = MagicMock()

        MoondreamActivityDetector("vikhyatk/moondream2", "2025-06-21")

        assert mock_model.call_args.kwargs["trust_remote_code"] is False


def test_moondream_remote_code_requires_commit_sha():
    with pytest.raises(ValueError, match="40-character commit SHA"):
        MoondreamActivityDetector(
            "vikhyatk/moondream2", "2025-06-21", allow_remote_code=True
        )


def test_moondream_remote_code_accepts_pinned_sha():
    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model,
        patch("transformers.AutoTokenizer.from_pretrained"),
    ):
        mock_model.return_value.to.return_value = MagicMock()

        MoondreamActivityDetector(
            "vikhyatk/moondream2", PINNED_REVISION, allow_remote_code=True
        )

        assert mock_model.call_args.kwargs["trust_remote_code"] is True
        assert mock_model.call_args.kwargs["revision"] == PINNED_REVISION


@pytest.mark.parametrize(
    "model_path",
    [
        "https://evil.example/model.pt",
        "http://evil.example/model.pt",
        "../model.pt",
        "nested/../../model.pt",
        "model.bin",
        "",
    ],
)
def test_rejects_unsafe_yolo_model_references(model_path):
    with pytest.raises(ValueError):
        validate_model_reference(model_path)


def test_accepts_relative_yolo_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model.pt").write_bytes(b"weights")
    assert validate_model_reference("model.pt") == "model.pt"


def test_rejects_symlink_escape(tmp_path, monkeypatch):
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"weights")
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "model.pt").symlink_to(outside)
    monkeypatch.chdir(workdir)

    with pytest.raises(ValueError):
        validate_model_reference("model.pt")
