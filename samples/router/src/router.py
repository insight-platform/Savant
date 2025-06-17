from time import time
from typing import Any

from savant_rs import register_handler
from savant_rs.logging import LogLevel, log
from savant_rs.utils.serialization import Message


class IngressHandler:
    """
    This handler is called for each message received from the ingress.
    """

    def __init__(self, period: int):
        self.period = period
        self.stream_screenshot_scheduler = {}

    def __call__(
        self, message_id: int, ingress_name: str, topic: str, message: Message
    ):
        log(
            LogLevel.Debug,
            "ingress_handler",
            f"Received message from {ingress_name} for topic {topic}",
        )
        # message.labels = [branches[0]]
        log(
            LogLevel.Debug,
            "ingress_handler",
            f"Sending message from {ingress_name} for topic {topic} with labels {message.labels}",
        )

        if message.is_video_frame():
            frame = message.as_video_frame()
            if not frame.keyframe:
                return message

            now = time()
            if topic not in self.stream_screenshot_scheduler:
                self.stream_screenshot_scheduler[topic] = 0
            if now - self.stream_screenshot_scheduler[topic] > self.period:
                self.stream_screenshot_scheduler[topic] = now
                # TODO: save screenshot
                log(LogLevel.Info, "ingress_handler", f"Marking a frame as a screenshot for {topic}")
                message.labels = ["screenshots"]

        return message


def init(params: Any):
    """
    This function is called once when the service starts. It is specified in the configuration.json file.
    """
    log(LogLevel.Info, "router::init", f"Initializing router service with params: {params}")
    screenshot_period = params.get("screenshot_period", 10)
    register_handler("ingress_handler", IngressHandler(screenshot_period))
    log(LogLevel.Info, "router::init", "Router service initialized successfully")
    return True