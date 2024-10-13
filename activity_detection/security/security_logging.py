import os
from abc import ABC, abstractmethod

from activity_detection.logging_config import setup_logger
from gtts import gTTS


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
        message = "Person detected!"
        self.logger.info(message)
        self.say_log(message)

    def log_no_suspicious_activity(self):
        message = "No persons detected."
        self.logger.info(message)
        self.say_log(message)

    def say_log(self, message):
        tts = gTTS(text=message, lang="en")
        tts.save("log_message.mp3")
        os.system("afplay log_message.mp3")
