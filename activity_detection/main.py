import yaml

from activity_detection.activity_manager import ActivityManager


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    activity_manager = ActivityManager.from_config(config)
    activity_manager.run_activity_detection()


if __name__ == "__main__":
    main()
