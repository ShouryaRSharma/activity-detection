from abc import ABC, abstractmethod

from activity_detection.logging_config import setup_logger


class SecurityLoggingInterface(ABC):
    @abstractmethod
    def log_suspicious_activity(self):
        pass

    @abstractmethod
    def log_no_suspicious_activity(self):
        pass


class DefaultSecurityLogging(SecurityLoggingInterface):
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)

    def log_suspicious_activity(self):
        self.logger.info("Suspicious activity detected!")

    def log_no_suspicious_activity(self):
        self.logger.info("No suspicious activity detected.")
