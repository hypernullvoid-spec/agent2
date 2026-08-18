"""
Notification Gateway for sending notifications.
"""

from typing import Any


class NotificationGateway:
    """Gateway for sending notifications to various destinations."""

    def __init__(self, config: Any):
        self.config = config
        self._enabled = config.enabled if config else False

    async def start(self):
        """Start the notification gateway."""
        pass

    async def close(self):
        """Close the notification gateway."""
        pass

    async def send(self, message: str, destinations: list[str] | None = None):
        """Send a notification."""
        if not self._enabled:
            return
        # Implementation would send to webhook, etc.
        pass