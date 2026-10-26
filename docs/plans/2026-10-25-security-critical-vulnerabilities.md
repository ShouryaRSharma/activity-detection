# Critical Security Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the critical security vulnerabilities in the activity detection application, dependencies, container image, and CI pipeline without losing existing functionality.

**Architecture:** Harden the model-loading and video-capture boundaries, make remote model code opt-in with immutable revisions, bound worker failures, upgrade to fixed dependency releases, and rebuild the container/CI pipeline around hashed installs and least-privilege test execution.

**Tech Stack:** Python 3.11, Poetry 2.5.1, pytest 9, ruff, torch 2.14, transformers 5.20, Ultralytics 8.4, Pillow 13, OpenCV 5, Docker, GitHub Actions.

---

## Task 1: Upgrade and clean the dependency declarations

**Files:**
- Modify: `pyproject.toml`

**Step 1: Replace the dependency block**

```toml
[tool.poetry.dependencies]
python = "^3.11"
transformers = "^5.20.2"
einops = "^0.8.2"
opencv-contrib-python-headless = "^5.0.0.93"
pillow = "^13.0.0"
torch = "^2.14.1"
torchvision = "^0.29.1"
ultralytics = "^8.4.188"
pyyaml = "^6.0.3"
numpy = "^2.4.6"

[tool.poetry.group.dev.dependencies]
ruff = "^0.17.2"
pytest = "^9.1.1"
pre-commit = "^4.7.0"
pytest-cov = "^7.1.0"
pip-audit = "^2.10.1"
```

Remove `setuptools` and `python-dotenv`. Both are unused.

**Step 2: Regenerate the lockfile**

Run:

```bash
uv tool run --python 3.11 --from poetry==2.5.1 \
  --with poetry-plugin-export==1.10.1 poetry lock
```

Expected: `poetry.lock` is rewritten and resolves successfully for Python 3.11.

**Step 3: Export audit requirements**

Run:

```bash
uv tool run --python 3.11 --from poetry==2.5.1 \
  --with poetry-plugin-export==1.10.1 \
  poetry export --without-hashes -f requirements.txt -o /tmp/activity-main.txt
uv tool run --python 3.11 --from poetry==2.5.1 \
  --with poetry-plugin-export==1.10.1 \
  poetry export --with dev --without-hashes -f requirements.txt \
  -o /tmp/activity-dev.txt
```

**Step 4: Audit the new lock**

Run:

```bash
uvx pip-audit -r /tmp/activity-main.txt --progress-spinner off
```

Expected: no critical or high vulnerabilities with an available fix. Record any
remaining advisories that have no fixed release.

**Step 5: Commit**

```bash
git add pyproject.toml poetry.lock
git commit -m "chore(deps): upgrade vulnerable dependencies"
```

---

## Task 2: Create the test environment

**Files:**
- No repository changes.

**Step 1: Create a Python 3.11 virtual environment**

```bash
uv venv --python 3.11 .venv
```

**Step 2: Install CPU-only torch for fast local testing**

```bash
uv pip install --python .venv/bin/python \
  torch==2.14.1 torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

**Step 3: Install the locked development requirements**

```bash
uv pip install --python .venv/bin/python -r /tmp/activity-dev.txt
```

**Step 4: Run the existing tests**

```bash
.venv/bin/python -m pytest -q
```

Expected: all existing tests pass or are explicitly xfailed. Investigate any
failure caused by the major upgrades before continuing.

---

## Task 3: Make Moondream remote code opt-in and pinned

**Files:**
- Modify: `activity_detection/classifiers/moondream_classifier.py`
- Test: `tests/test_activity_classifier.py`

**Step 1: Write the failing tests**

Add to `tests/test_activity_classifier.py`:

```python
import re


PINNED_REVISION = "a" * 40


def test_moondream_disables_remote_code_by_default():
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained"
    ) as mock_model, patch("transformers.AutoTokenizer.from_pretrained"):
        mock_model.return_value.to.return_value = MagicMock()

        MoondreamActivityDetector("vikhyatk/moondream2", "2025-06-21")

        assert mock_model.call_args.kwargs["trust_remote_code"] is False


def test_moondream_remote_code_requires_commit_sha():
    with pytest.raises(ValueError, match="40-character commit SHA"):
        MoondreamActivityDetector(
            "vikhyatk/moondream2", "2025-06-21", allow_remote_code=True
        )


def test_moondream_remote_code_accepts_pinned_sha():
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained"
    ) as mock_model, patch("transformers.AutoTokenizer.from_pretrained"):
        mock_model.return_value.to.return_value = MagicMock()

        MoondreamActivityDetector(
            "vikhyatk/moondream2", PINNED_REVISION, allow_remote_code=True
        )

        assert mock_model.call_args.kwargs["trust_remote_code"] is True
        assert mock_model.call_args.kwargs["revision"] == PINNED_REVISION
```

Update the existing parameterized tests so Moondream is constructed with
`allow_remote_code=True` and `PINNED_REVISION`:

```python
detector = detector_class(
    "vikhyatk/moondream2", PINNED_REVISION, allow_remote_code=True
)
```

**Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_activity_classifier.py -q
```

Expected: `TypeError` because `allow_remote_code` is not accepted.

**Step 3: Implement the change**

Replace the constructor in
`activity_detection/classifiers/moondream_classifier.py`:

```python
import re

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class MoondreamActivityDetector(ActivityDetectionInterface):
    def __init__(
        self,
        model_id: str,
        revision: str,
        allow_remote_code: bool = False,
    ):
        super().__init__()
        if allow_remote_code and not _COMMIT_SHA.fullmatch(revision):
            raise ValueError(
                "A full 40-character commit SHA revision is required when "
                "allow_remote_code is enabled"
            )
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=allow_remote_code,
            revision=revision,
            dtype=torch.float32,
        ).to(self.device.value)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            use_fast=True,
            trust_remote_code=allow_remote_code,
        )
```

**Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_activity_classifier.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add activity_detection/classifiers/moondream_classifier.py \
  tests/test_activity_classifier.py
git commit -m "fix(security): require opt-in and pinned revision for remote model code"
```

---

## Task 4: Validate YOLO model references

**Files:**
- Modify: `activity_detection/classifiers/yolo_classifiers.py`
- Test: `tests/test_activity_classifier.py`

**Step 1: Write the failing tests**

Add to `tests/test_activity_classifier.py`:

```python
from activity_detection.classifiers.yolo_classifiers import (
    validate_model_reference,
)


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
```

**Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_activity_classifier.py -q
```

Expected: `ImportError` for `validate_model_reference`.

**Step 3: Implement the validator**

Add to `activity_detection/classifiers/yolo_classifiers.py`:

```python
from pathlib import Path


def validate_model_reference(model_path: str) -> str:
    """Reject remote, traversal, and non-PyTorch model references."""
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("model_path must be a non-empty string")
    if "://" in model_path:
        raise ValueError("Remote model URLs are not allowed")

    path = Path(model_path)
    if ".." in path.parts:
        raise ValueError("model_path must not contain '..'")
    if path.suffix.lower() != ".pt":
        raise ValueError("Only .pt model files are supported")

    if path.is_absolute():
        if not path.is_file():
            raise ValueError("Absolute model paths must point to an existing file")
        return str(path)

    cwd = Path.cwd().resolve()
    resolved = (cwd / path).resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError as exc:
        raise ValueError("Model path escapes the working directory") from exc
    if resolved.exists() and not resolved.is_file():
        raise ValueError("Model path must point to a file")
    return model_path
```

Call it from `BaseYOLODetector.__init__`:

```python
self.model = self.load_model(validate_model_reference(model_path)).to(
    self.device.value
)
```

**Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_activity_classifier.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add activity_detection/classifiers/yolo_classifiers.py \
  tests/test_activity_classifier.py
git commit -m "fix(security): validate YOLO model references"
```

---

## Task 5: Harden video capture

**Files:**
- Modify: `activity_detection/security/security_capture.py`
- Create: `tests/test_security_capture.py`
- Modify: `tests/test_security_module.py`

**Step 1: Write the failing tests**

Create `tests/test_security_capture.py`:

```python
from unittest.mock import patch

import numpy as np
import pytest

from activity_detection.security.security_capture import DefaultVideoCapture


class FakeWriter:
    instances = []

    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.size = size
        self.frames = []
        self.released = False
        FakeWriter.instances.append(self)

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


@pytest.fixture(autouse=True)
def reset_fake_writers():
    FakeWriter.instances.clear()


def test_capture_writes_into_output_dir_with_frame_size(tmp_path):
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    capture = DefaultVideoCapture(str(tmp_path / "clips"))

    with patch(
        "activity_detection.security.security_capture.cv2.VideoWriter",
        FakeWriter,
    ), patch(
        "activity_detection.security.security_capture.cv2.VideoWriter.fourcc",
        return_value=0,
    ):
        capture.start_video_capture()
        capture.capture_frame(frame)
        capture.stop_video_capture()

    writer = FakeWriter.instances[0]
    assert writer.size == (20, 10)
    assert writer.frames == [frame]
    assert writer.released is True
    assert str(tmp_path / "clips") in writer.path


def test_capture_rotates_after_max_clip_seconds(tmp_path):
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    capture = DefaultVideoCapture(
        str(tmp_path / "clips"), max_clip_seconds=1.0
    )

    with patch(
        "activity_detection.security.security_capture.cv2.VideoWriter",
        FakeWriter,
    ), patch(
        "activity_detection.security.security_capture.cv2.VideoWriter.fourcc",
        return_value=0,
    ), patch(
        "activity_detection.security.security_capture.time.monotonic",
        side_effect=[0.0, 0.5, 2.0],
    ):
        capture.start_video_capture()
        capture.capture_frame(frame)
        capture.capture_frame(frame)
        capture.stop_video_capture()

    assert len(FakeWriter.instances) == 2
    assert FakeWriter.instances[0].released is True


def test_stop_without_start_is_safe(tmp_path):
    capture = DefaultVideoCapture(str(tmp_path / "clips"))
    capture.stop_video_capture()
```

Replace the stale xfail tests in `tests/test_security_module.py` with:

```python
import numpy as np

from activity_detection.classifiers.types import Prediction
from activity_detection.security.security_module import SecurityModule


class FakeCapture:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.frames = []

    def start_video_capture(self):
        self.started += 1

    def stop_video_capture(self):
        self.stopped += 1

    def capture_frame(self, frame):
        self.frames.append(frame)


class FakeLogging:
    def __init__(self):
        self.suspicious = 0
        self.clear = 0

    def log_suspicious_activity(self):
        self.suspicious += 1

    def log_no_suspicious_activity(self):
        self.clear += 1


def test_security_module_starts_and_stops_capture():
    capture = FakeCapture()
    logging = FakeLogging()
    module = SecurityModule(capture, logging, stop_threshold=2)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    module.process_frame(frame, Prediction(detected=True))
    module.process_frame(frame, Prediction(detected=False))
    module.process_frame(frame, Prediction(detected=False))

    assert capture.started == 1
    assert capture.stopped == 1
    assert logging.suspicious == 1
    assert logging.clear == 1
    assert len(capture.frames) == 2
```

**Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_security_capture.py \
  tests/test_security_module.py -q
```

Expected: FAIL because the writer ignores `output_dir` and uses a fixed size.

**Step 3: Implement the hardened capture class**

Replace `DefaultVideoCapture` in
`activity_detection/security/security_capture.py`:

```python
import datetime
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from activity_detection.logging_config import setup_logger


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
                and time.monotonic() - self.clip_started_at
                >= self.max_clip_seconds
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
        self.output_file = (
            self.output_dir / f"suspicious_activity_{timestamp}.mp4"
        )
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            str(self.output_file), fourcc, self.fps, (width, height)
        )
        if not self.video_writer.isOpened():
            self.video_writer = None
            raise ValueError(
                f"Failed to open video writer for {self.output_file}"
            )
        self.clip_started_at = time.monotonic()
        self.logger.info("Video capture started: %s", self.output_file)

    def _release_writer(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            if self.output_file is not None:
                self.logger.info("Video saved: %s", self.output_file)
```

**Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_security_capture.py \
  tests/test_security_module.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add activity_detection/security/security_capture.py \
  tests/test_security_capture.py tests/test_security_module.py
git commit -m "fix(security): bound video capture output and honor output directory"
```

---

## Task 6: Stop worker threads safely

**Files:**
- Modify: `activity_detection/activity_manager.py`
- Test: `tests/test_activity_manager.py`

**Step 1: Write the failing test**

Create `tests/test_activity_manager.py`:

```python
import pytest

from activity_detection.activity_manager import ActivityManager


class FailingCamera:
    def start_capture(self):
        pass

    def stop_capture(self):
        pass

    def get_frame(self):
        raise IOError("camera offline")


class PassthroughProcessor:
    def process_frame(self, frame):
        return frame


class NoopDetector:
    def detect_activity(self, image):
        raise AssertionError("detector should not be reached")


class NoopVideoCapture:
    def stop_video_capture(self):
        pass


class NoopSecurityModule:
    video_capture = NoopVideoCapture()

    def process_frame(self, frame, prediction):
        pass


def test_worker_failure_stops_and_raises():
    manager = ActivityManager(
        FailingCamera(),
        PassthroughProcessor(),
        NoopDetector(),
        NoopSecurityModule(),
    )

    with pytest.raises(RuntimeError, match="worker thread"):
        manager.run_activity_detection()
```

**Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_activity_manager.py -q
```

Expected: the test hangs or fails because the failure is swallowed and the
process keeps running.

**Step 3: Implement the stop event and error propagation**

In `activity_detection/activity_manager.py`, replace the `running` flag and the
three worker methods and `run_activity_detection` with:

```python
        self.stop_event = threading.Event()
        self._thread_errors: list[BaseException] = []
        self._error_lock = threading.Lock()

    def _record_error(self, exc: BaseException):
        with self._error_lock:
            self._thread_errors.append(exc)
        self.logger.exception("Worker thread failed: %s", exc)
        self.stop_event.set()

    def capture_frames(self):
        try:
            self.camera_input.start_capture()
            while not self.stop_event.is_set():
                frame = self.camera_input.get_frame()
                try:
                    self.frame_queue.put(frame, timeout=0.5)
                except Full:
                    self.logger.warning("Frame queue is full. Skipping frame.")
        except Exception as exc:
            self._record_error(exc)

    def process_frames(self):
        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                except Empty:
                    continue
                processed_image = self.image_processor.process_frame(frame)
                prediction = self.activity_detector.detect_activity(
                    processed_image
                )
                try:
                    self.processed_frame_queue.put(
                        (frame, prediction), timeout=0.5
                    )
                except Full:
                    self.logger.warning(
                        "Processed frame queue is full. Skipping frame."
                    )
        except Exception as exc:
            self._record_error(exc)

    def write_video(self):
        try:
            while not self.stop_event.is_set():
                try:
                    frame, prediction = self.processed_frame_queue.get(
                        timeout=0.5
                    )
                except Empty:
                    continue
                self.security_module.process_frame(frame, prediction)
        except Exception as exc:
            self._record_error(exc)

    def run_activity_detection(self):
        threads = [
            threading.Thread(target=self.capture_frames, daemon=True),
            threading.Thread(target=self.process_frames, daemon=True),
            threading.Thread(target=self.write_video, daemon=True),
        ]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user. Stopping the program.")
        finally:
            self.stop_event.set()
            self.camera_input.stop_capture()
            self.security_module.video_capture.stop_video_capture()
            for thread in threads:
                if thread.is_alive():
                    thread.join(timeout=5)
            self.logger.info("Activity detection stopped.")

        if self._thread_errors:
            raise RuntimeError("One or more worker threads failed") from (
                self._thread_errors[0]
            )
```

Remove all uses of `self.running`.

**Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_activity_manager.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add activity_detection/activity_manager.py tests/test_activity_manager.py
git commit -m "fix: stop worker threads and surface capture failures"
```

---

## Task 7: Make logger setup idempotent

**Files:**
- Modify: `activity_detection/logging_config.py`
- Test: `tests/test_logging_config.py`

**Step 1: Write the failing test**

Create `tests/test_logging_config.py`:

```python
from activity_detection.logging_config import setup_logger


def test_setup_logger_does_not_duplicate_handlers():
    name = "test_idempotent_logger"
    first = setup_logger(name)
    handler_count = len(first.handlers)

    second = setup_logger(name)

    assert second is first
    assert len(second.handlers) == handler_count
```

**Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_logging_config.py -q
```

Expected: FAIL because a second handler is added.

**Step 3: Implement the change**

```python
def setup_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
```

**Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_logging_config.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add activity_detection/logging_config.py tests/test_logging_config.py
git commit -m "fix: avoid duplicate log handlers"
```

---

## Task 8: Harden the Docker build

**Files:**
- Modify: `Dockerfile`
- Create: `.dockerignore`

**Step 1: Replace the Dockerfile**

```dockerfile
ARG POETRY_VERSION=2.5.1
ARG POETRY_EXPORT_VERSION=1.10.1

FROM python:3.11-slim-bookworm AS base

ARG POETRY_VERSION
ARG POETRY_EXPORT_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install \
    "poetry==${POETRY_VERSION}" \
    "poetry-plugin-export==${POETRY_EXPORT_VERSION}"

WORKDIR /temp

COPY ./pyproject.toml /temp/pyproject.toml
COPY ./poetry.lock /temp/poetry.lock

RUN poetry config virtualenvs.create false \
    && poetry export --without-urls -f requirements.txt \
        --output requirements.txt \
    && poetry export --without-urls --with dev -f requirements.txt \
        --output requirements-dev.txt

FROM python:3.11-slim-bookworm AS app-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

RUN addgroup --system activity_detection \
    && adduser --system --ingroup activity_detection activity_detection

FROM app-base AS test

COPY --from=base /temp/requirements-dev.txt /code/requirements-dev.txt

RUN pip install --require-hashes -r requirements-dev.txt

COPY --chown=activity_detection:activity_detection ./activity_detection /code/activity_detection
COPY --chown=activity_detection:activity_detection ./tests /code/tests
COPY --chown=activity_detection:activity_detection ./pyproject.toml /code/pyproject.toml
COPY --chown=activity_detection:activity_detection ./poetry.lock /code/poetry.lock

USER activity_detection

ENTRYPOINT ["python", "-m", "activity_detection.main"]

FROM app-base AS production

COPY --from=base /temp/requirements.txt /code/requirements.txt

RUN pip install --require-hashes -r requirements.txt

COPY --chown=activity_detection:activity_detection ./activity_detection /code/activity_detection
COPY --chown=activity_detection:activity_detection ./pyproject.toml /code/pyproject.toml
COPY --chown=activity_detection:activity_detection ./poetry.lock /code/poetry.lock

USER activity_detection

ENTRYPOINT ["python", "-m", "activity_detection.main"]
```

**Step 2: Create `.dockerignore`**

```text
.git
.github
.env
.venv
**/__pycache__
*.pyc
*.mp4
*.pt
*.pth
*.onnx
*.engine
*.db
config.yaml
test-reports
.coverage
coverage.xml
docs
```

**Step 3: Validate the Dockerfile when Docker is available**

```bash
docker build --target test -t activity-detection:test .
```

Expected: image builds and hashed requirements install successfully. If Docker is
not available locally, note that CI must validate this step.

**Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "chore(security): harden container build and restrict build context"
```

---

## Task 9: Harden CI

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1: Replace the workflow**

```yaml
name: CI with Docker

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build test image
        run: docker build --target test -t activity-detection:test .

      - name: Run tests
        run: |
          mkdir -p test-reports
          chmod 777 test-reports
          docker run --rm \
            -v "$PWD/test-reports:/code/test-reports" \
            --entrypoint "" \
            activity-detection:test \
            python -m pytest --cov=. \
              --cov-report=xml:test-reports/coverage.xml \
              --cov-report=term \
              --cov-report=html:test-reports/htmlcov \
              --junitxml=test-reports/junit.xml

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: test-reports

  publish:
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_ACCESS_TOKEN }}

      - name: Build and push production image
        uses: docker/build-push-action@v6
        with:
          context: .
          target: production
          push: true
          tags: ${{ secrets.DOCKER_USERNAME }}/my-app:latest
```

**Step 2: Validate the YAML**

```bash
uvx --from yamllint yamllint .github/workflows/ci.yml
```

Expected: no syntax errors. If `yamllint` is unavailable, use a YAML parser:

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "chore(ci): remove PR registry credentials and deprecated actions"
```

---

## Task 10: Update documentation

**Files:**
- Modify: `README.md`

**Step 1: Update the Moondream configuration example**

```yaml
activity_detector:
  # Pick one of the below
  YOLOActivityDetector:
    args:
      - 0.9 # Confidence threshold
  YOLOWorldActivityDetector:
    args:
      - 0.9 # Confidence threshold
  MoondreamActivityDetector:
    args:
      - "vikhyatk/moondream2" # Model name
      - "5d6c926f44e26b07957b0dd315bbedcb4c17a5fe" # Pinned commit SHA
      - true # Explicitly allow remote model code
```

Add a short warning after the example:

```markdown
`MoondreamActivityDetector` loads custom model code. Remote code is disabled by
default. If you enable it, use a full 40-character commit SHA as the revision.
Only load models you trust.
```

**Step 2: Update the Docker run example**

```bash
docker run \
  -v "$(pwd)/config.yaml:/code/config.yaml:ro" \
  activity-detection
```

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document pinned remote model code and container config mount"
```

---

## Task 11: Full verification

**Files:**
- No changes.

**Step 1: Run the full test suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

**Step 2: Run lint and formatting checks**

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Expected: both commands pass. If formatting changes are needed, run
`.venv/bin/ruff format .` and re-run the checks.

**Step 3: Re-run the dependency audit**

```bash
uvx pip-audit -r /tmp/activity-dev.txt --progress-spinner off
```

Expected: no critical or high vulnerabilities with an available fix. Document
any remaining no-fix advisories.

**Step 4: Check the diff**

```bash
git status
git diff --stat main...HEAD
git log --oneline main..HEAD
```

**Step 5: Final commit if verification produced changes**

```bash
git add -A
git commit -m "test: verify security remediation"
```

Do not create an empty commit if there are no changes.

---

## Residual risk checklist

- [ ] Any dependency advisory without a fixed release is documented.
- [ ] The Moondream opt-in requirement is visible in the README.
- [ ] The video capture rotation interval is documented or configurable.
- [ ] CI passes on a real Docker-capable runner.
- [ ] Operators monitor disk usage if recordings are retained indefinitely.
