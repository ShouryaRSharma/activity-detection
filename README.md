# Suspicious Activity Detection System

The Suspicious Activity Detection System (SADS) is a Python application that monitors a live video stream from a camera,
detects suspicious activity using some type of classification model, and logs security events with a captured video and
console logs.

## TODO

- [ ] Evaluate the models for person detection (we have no ground truth yet lol)
- [ ] Add background task of summarising captured video
- [ ] Add email client to send captured video to end user

## Architecture Diagram

[architecture-diagram]: ./docs/architectureDiagram.svg

![Architecture Diagram][architecture-diagram]

## Key Features

- Real-time video stream processing from an IP camera
- Person detection using the Moondream image-text vision model or YoloWorld object detection model
- Video capture and security logging triggered by suspicious activity
- Extensible design for adding new detection criteria and logging mechanisms

## License

This project is licensed under the MIT License.

## Installation

### Prerequisites

- Python 3.11 through 3.13
- Poetry 2.x
- Docker

### Setup

1. Clone the repository
2. Install dependencies using Poetry:

    ```bash
    poetry install
    ```

3. Copy `config.yaml.template` to `config.yaml` and edit it with the following settings:

    ```yaml
    camera_input:
      # Pick one of the below
      IPCamera:
        args:
          - "<IPV4_ADDRESS>"
      # LocalCamera:
      #   args:
      #     - 0

    image_processor:
      DefaultImageProcessor: {}

    activity_detector:
      # Pick one of the below
      YOLOActivityDetector:
        args:
          - "yolo11m.pt" # Local model file
          - 0.9 # Confidence threshold
      # YOLOWorldActivityDetector:
      #   args:
      #     - "yolov8m-worldv2.pt" # Local model file
      #     - 0.9 # Confidence threshold
      # MoondreamActivityDetector:
      #   args:
      #     - "vikhyatk/moondream2" # Model name
      #     - "5d6c926f44e26b07957b0dd315bbedcb4c17a5fe" # Pinned commit SHA
      #     - true # Explicitly allow remote model code

    security_logging:
      DefaultSecurityLogging: {}

    video_capture:
      DefaultVideoCapture:
        args:
          - "suspicious_activity_videos"

    ```

`MoondreamActivityDetector` loads custom model code. Remote code is disabled by
default. If you enable it, use a full 40-character commit SHA as the revision.
Only load models you trust.

`YOLOActivityDetector` only accepts local `.pt` files and rejects URLs, path
traversal, and symlinks that escape the working directory.

## Usage

For local development, you can run the application using the following command:

```bash
poetry run activity-detection
```

To run the application in a Docker container, mount your configuration and
recordings directory:

```bash
docker build -t activity-detection .
docker run --rm \
  -v "$(pwd)/config.yaml:/app/config.yaml:ro" \
  -v "$(pwd)/suspicious_activity_videos:/app/suspicious_activity_videos" \
  activity-detection
```

## Testing

To run the test suite, use the following command:

```bash
poetry run pytest
```
