from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AwsCredentials(BaseModel):
    """AWS credentials."""

    region: str = Field(description='AWS region.')
    access_key: str = Field(description='AWS access key ID.')
    secret_key: str = Field(description='AWS secret access key.')


class StreamModel(BaseModel):
    """Stream configuration."""

    name: Optional[str] = Field(None, description='KVS stream name.')
    source_id: Optional[str] = Field(None, description='Source ID for the stream.')
    timestamp: Optional[datetime] = Field(
        None,
        description='Where to start reading the stream.',
    )
    credentials: Optional[AwsCredentials] = Field(
        None,
        description='AWS credentials for KVS stream.',
    )
    is_playing: Optional[bool] = Field(
        None,
        description='Whether the stream is playing.',
    )

    def without_credentials(self):
        """Return a copy of the stream configuration without credentials."""

        return StreamModel(
            name=self.name,
            source_id=self.source_id,
            timestamp=self.timestamp,
            is_playing=self.is_playing,
        )
