from __future__ import annotations as _annotations

from datetime import datetime

from pydantic import AwareDatetime, BaseModel, Field


class TimeRangeBuilderSuccess(BaseModel, use_attribute_docstrings=True):
    """Response when a time range could be successfully generated."""

    start_timestamp: AwareDatetime = Field(serialization_alias='startTimestamp')
    """A datetime in ISO format with timezone offset when the interval starts."""

    end_timestamp: AwareDatetime = Field(serialization_alias='endTimestamp')
    """A datetime in ISO format with timezone offset when the interval ends."""

    explanation: str | None
    """
    A brief explanation of the time range that was selected.

    For example, if a user only mentions a specific point in time, you might explain that you selected a 10 minute
    window around that time.
    """


class TimeRangeBuilderError(BaseModel):
    """Response when a time range cannot not be generated."""

    error: str


TimeRangeResponse = TimeRangeBuilderSuccess | TimeRangeBuilderError


class TimeRangeInputs(BaseModel):
    """The inputs for the time range inference agent."""

    prompt: str
    now: AwareDatetime = Field(default_factory=lambda: datetime.now().astimezone())
