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

- Python 3.11 or higher
- Poetry 1.8 or higher
- Docker

### Setup

1. Clone the repository
2. Install dependencies using Poetry:

    ```bash
    poetry install
    ```

3. Create a `config.yaml` file in the project root directory with the following environment variables:

    ```yaml
    camera_input:
      # Pick one of the below
      IPCamera:
        args:
          - "<IPV4_ADDRESS>"
      LocalCamera:
        args:
          - 0

    image_processor:
      DefaultImageProcessor: {}

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
          - "2024-08-26" # Revision date

    security_logging:
      DefaultSecurityLogging: {}

    video_capture:
      DefaultVideoCapture:
        args:
          - "suspicious_activity_videos"

    ```

## Usage

For local development, you can run the application using the following command:

```bash
poetry run activity-detection
```

To run the application in a Docker container, use the following command:

```bash
docker build -t activity-detection .
docker run activity-detection
```

## Testing

To run the test suite, use the following command:

```bash
poetry run pytest
```
