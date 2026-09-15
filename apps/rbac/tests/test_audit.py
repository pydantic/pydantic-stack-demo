"""Offline tests against responses shaped like the Platform API OpenAPI schemas."""

import json
from dataclasses import asdict

import httpx
import logfire

from rbac import audit

logfire.configure(send_to_logfire=False, console=False)

PROJECT_A = '11111111-1111-1111-1111-111111111111'
PROJECT_B = '22222222-2222-2222-2222-222222222222'

FIXTURES: dict[str, object] = {
    '/api/v1/organization/': {
        'id': 'org-1',
        'organization_name': 'acme',
        'organization_display_name': 'Acme',
        'subscription_plan': 'enterprise',
        'gateway_enabled': True,
        'created_at': '2025-01-01T00:00:00Z',
        'updated_at': '2025-01-01T00:00:00Z',
    },
    '/api/v1/projects/': [
        {
            'id': PROJECT_A,
            'project_name': 'prod',
            'organization_name': 'acme',
            'visibility': 'private',
            'created_at': '',
            'updated_at': '',
        },
        {
            'id': PROJECT_B,
            'project_name': 'staging',
            'organization_name': 'acme',
            'visibility': 'public',
            'created_at': '',
            'updated_at': '',
        },
    ],
    '/api/v1/members/': [
        {'id': 'u1', 'email': 'a@acme.example', 'name': 'A', 'role': 'admin', 'role_id': 'r1', 'member_since': ''},
        {'id': 'u2', 'email': 'b@acme.example', 'name': 'B', 'role': 'admin', 'role_id': 'r1', 'member_since': ''},
        {'id': 'u3', 'email': 'c@acme.example', 'name': 'C', 'role': 'member', 'role_id': 'r2', 'member_since': ''},
        {'id': 'u4', 'email': 'g@partner.example', 'name': 'G', 'role': 'guest', 'role_id': 'r3', 'member_since': ''},
    ],
    '/api/v1/invitations/': [
        {
            'id': 'inv-1',
            'created_at': '',
            'expiration': None,
            'invite_url': 'x',
            'max_usage_count': 5,
            'organization_id': 'org-1',
            'role_id': 'r2',
            'usage_count': 0,
            'last_used_at': None,
        }
    ],
    '/api/v1/api-keys/': [
        {
            'id': 'k1',
            'name': 'ci-deploy',
            'active': True,
            'all_projects': True,
            'expires_at': None,
            'last_used_at': None,
            'scopes': ['project:read'],
            'organization_id': 'org-1',
            'created_at': '',
            'claims': {},
        },
        {
            'id': 'k2',
            'name': 'dashboards',
            'active': True,
            'all_projects': False,
            'project_name': 'prod',
            'expires_at': '2999-01-01T00:00:00Z',
            'scopes': ['project:read_otlp'],
            'organization_id': 'org-1',
            'created_at': '',
            'claims': {},
        },
    ],
    f'/api/v1/projects/{PROJECT_A}/write-tokens/': [
        {
            'id': 't1',
            'token_prefix': 'pylf_v1_us_ab',
            'active': True,
            'expires_at': '2020-01-01T00:00:00Z',
            'created_at': '2019-01-01T00:00:00Z',
            'project_id': PROJECT_A,
            'project_name': 'prod',
            'created_by_name': 'A',
            'frontend_application_deleted': False,
        },
    ],
    f'/api/v1/projects/{PROJECT_A}/read-tokens/': [],
    f'/api/v1/projects/{PROJECT_B}/write-tokens/': [],
    f'/api/v1/projects/{PROJECT_B}/read-tokens/': [
        {
            'id': 't2',
            'token_prefix': 'pylf_v1_us_cd',
            'active': True,
            'expires_at': '2999-01-01T00:00:00Z',
            'created_at': '',
            'project_id': PROJECT_B,
            'project_name': 'staging',
            'created_by_name': 'B',
        },
    ],
}


def handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == '/api/v1/gateway/providers/':
        return httpx.Response(403, json={'detail': 'This token is missing the required scope: organization:read.'})
    if path in FIXTURES:
        return httpx.Response(200, json=FIXTURES[path])
    return httpx.Response(404, json={'detail': 'Not Found'})


def make_client() -> audit.PlatformClient:
    return audit.PlatformClient(
        'pylf_v1_us_test', 'https://api-us.pydantic.dev', transport=httpx.MockTransport(handler)
    )


def test_collect_and_findings():
    report = audit.collect(make_client(), max_admins=1)
    messages = [f.message for f in report.findings]
    assert len(report.projects) == 2 and len(report.members) == 4
    assert any('2 organization admins' in m for m in messages)
    assert any('guest' in m for m in messages)
    assert any('inv-1' in m for m in messages)
    assert any('ci-deploy' in m and 'never expires' in m for m in messages)
    assert any('expired on 2020-01-01' in m for m in messages)
    assert any('staging has no active write token' in m for m in messages)
    assert any('/v1/gateway/providers/' in m and 'organization:read' in m for m in messages)
    assert report.gateway_providers == []


def test_render_and_json_roundtrip():
    report = audit.collect(make_client(), max_admins=3)
    text = audit.render(report)
    assert 'Organization: Acme' in text and 'Findings' in text
    payload = json.loads(json.dumps(asdict(report), default=str))
    assert payload['organization']['organization_name'] == 'acme'


def test_unauthorized_is_reported_not_raised():
    def deny(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={'detail': 'Could not validate credentials'})

    client = audit.PlatformClient('bad', 'https://api-us.pydantic.dev', transport=httpx.MockTransport(deny))
    report = audit.collect(client, max_admins=3)
    assert report.organization is None and report.projects == []
    assert all(f.status == 401 for f in report.unreadable)
    assert {f.scope for f in report.unreadable} <= set(audit.REQUIRED_SCOPES)


def test_main_without_key_explains_scopes(monkeypatch, capsys):
    monkeypatch.delenv('LOGFIRE_API_KEY', raising=False)
    assert audit.main([]) == 2
    err = capsys.readouterr().err
    assert 'organization:read_member' in err
