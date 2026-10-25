# Critical Security Remediation Design

Date: 2026-10-25
Branch: `security/critical-vulnerabilities`

## Goal

Remove critical and high-severity security risks from the activity detection
application, its dependency tree, container image, and CI pipeline while
preserving the existing detection functionality.

## Findings

### Application

- `MoondreamActivityDetector` loads Hugging Face model repositories with
  `trust_remote_code=True`, allowing arbitrary Python execution from the model
  repository.
- YOLO weights are loaded from a configuration-provided reference without
  validation, allowing URL-based or path-traversal references.
- `DefaultVideoCapture` ignores its configured output directory, writes to the
  current working directory, uses a hard-coded frame size, and records without a
  duration or size limit. Continuous detections can exhaust disk space.
- Worker threads catch no exceptions and use a bare `running` flag. A failed
  capture or inference thread can leave the process running and blocked.
- `setup_logger` adds a new console handler every time a logger is created.

### Dependencies

The locked dependency set is from late 2024 and contains known critical
advisories, including:

- `torch 2.3.1`: unsafe deserialization and memory-safety issues.
- `transformers 4.45.2`: remote code execution and deserialization issues.
- `pillow 10.4.0`: numerous image-parsing memory-safety and denial-of-service
  issues.
- `setuptools 70.3.0`: path traversal / code execution issues (unused by the
  application).
- Vulnerable transitive packages including `requests`, `urllib3`, `filelock`,
  `fsspec`, and `virtualenv`.

### Container and CI

- The production base image is `python:3.11.4-slim-buster`, based on an
  end-of-life Debian release.
- Dependencies are installed without hash verification.
- The Docker build context is not restricted, so `.env`, `.git`, recordings, and
  model weights can be sent to the Docker daemon.
- CI uses deprecated actions and the unmaintained
  `addnab/docker-run-action@v3`.
- Every pull request logs in to Docker Hub and pushes an image using registry
  credentials, even though tests do not require a registry.

## Design

### Dependency updates

- Remove `setuptools` and `python-dotenv`; both are unused at runtime.
- Declare `numpy` directly because the application imports it.
- Upgrade to the latest releases that support Python 3.11:
  - `torch 2.14.1`
  - `torchvision 0.29.1`
  - `transformers 5.20.2`
  - `pillow 13.0.0`
  - `ultralytics 8.4.188`
  - `opencv-contrib-python-headless 5.0.0.93`
  - `pyyaml 6.0.3`
  - `einops 0.8.2`
  - `pytest 9.1.1`, `pytest-cov 7.1.0`, `ruff 0.17.2`,
    `pre-commit 4.7.0`
- Regenerate `poetry.lock` with Poetry 2.5.1.
- Verify the result with `pip-audit`.

The latest Moondream 2 revision includes explicit support for
`transformers>=5`, so the major upgrade is expected to remain compatible.

### Moondream remote code

`MoondreamActivityDetector` gains an `allow_remote_code` argument that defaults
to `False`. Remote model code is only permitted when the caller explicitly opts
in, and the revision must be a full 40-character commit SHA. This prevents
mutable tags or branches from changing the code that is executed. The README
example is updated to use a pinned commit SHA.

### YOLO model references

A shared validator rejects model references that contain URL schemes, path
traversal, or unsafe extensions before the model is constructed. The default
official model filename remains supported.

### Video capture

`DefaultVideoCapture` will:

- Resolve and create the configured output directory.
- Generate unique filenames so two captures in the same second do not collide.
- Open the writer lazily using the first frame's dimensions.
- Rotate to a new clip after a configurable maximum duration, bounding disk
  usage for long-running detections.
- Always release the writer on stop and after errors.

### Worker lifecycle

`ActivityManager` uses a `threading.Event` for shutdown. Queue reads use short
timeouts so workers can observe the stop event. Exceptions in capture, inference,
or writing are logged and trigger shutdown instead of leaving the main thread
blocked forever.

### Logging

`setup_logger` becomes idempotent so repeated component construction does not
duplicate log lines.

### Container

- Use `python:3.11-slim-bookworm`.
- Remove the unused `curl` installation and Poetry from the runtime stages.
- Export locked requirements with hashes and install with `--require-hashes`.
- Create the non-root user before copying files and use `COPY --chown`.
- Add `.dockerignore` for secrets, VCS data, recordings, models, caches, and test
  reports.

### CI

- Update actions to current major versions.
- Remove the third-party Docker run action in favour of `docker run`.
- Do not authenticate to or push to Docker Hub on pull requests.
- Run tests on the locally built test image.
- Publish an image only for pushes to `main`.

## Testing

- Unit tests for remote-code opt-in and revision pinning.
- Unit tests for YOLO reference validation.
- Unit tests for output directory handling, frame-sized writers, and clip
  rotation.
- Existing camera, image-processing, and classifier tests must continue to pass.
- `ruff check` and `ruff format --check`.
- `pip-audit` against the regenerated lock.
- Docker image build and CI syntax validation are limited in this environment
  because Docker is not installed; CI must confirm the final image build.

## Residual Risk

Some advisories may not have a fixed release. Any remaining findings after the
upgrade will be documented with their applicability to this application.
Remote model code remains inherently dangerous even when pinned; the opt-in
flag is a deliberate trust boundary.
