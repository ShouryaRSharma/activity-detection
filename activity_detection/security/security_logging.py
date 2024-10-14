from abc import ABC, abstractmethod

from activity_detection.logging_config import setup_logger


class SecurityLoggingInterface(ABC):
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)

    @abstractmethod
    def log_suspicious_activity(self):
        pass

    @abstractmethod
    def log_no_suspicious_activity(self):
        pass


class DefaultSecurityLogging(SecurityLoggingInterface):
    def log_suspicious_activity(self):
        message = "Person detected!"
        self.logger.info(message)

    def log_no_suspicious_activity(self):
        message = "No persons detected."
        self.logger.info(message)
