"""Typed inputs, outputs and case metadata shared by the agent, the API and the evals."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SupportRequest(BaseModel):
    """What the service receives: a case id and the free-text request."""

    case_id: str
    request: str


class SupportResolution(BaseModel):
    """What the agent returns. `escalated` must mirror whether it queued an escalation."""

    resolution: str = Field(description='A concrete, safe next step an authorized operator can perform.')
    escalated: bool = Field(default=False, description='True only if queue_urgent_escalation was called.')


class CaseMetadata(BaseModel):
    """Why a case is in the regression dataset. Every case names the failure it guards against."""

    failure_mode: str
    expects_escalation: bool = False
    human_label: str = 'unreviewed'
    regression_candidate: bool = True
