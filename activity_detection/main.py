import yaml

from pathlib import Path
from activity_detection.activity_manager import ActivityManager


def main():
    current_dir = Path(__file__).parent.parent
    config_file = current_dir / "config.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    activity_manager = ActivityManager.from_config(config)
    activity_manager.run_activity_detection()


if __name__ == "__main__":
    main()
