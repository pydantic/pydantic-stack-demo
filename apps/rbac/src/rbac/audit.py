"""Audit who can see and send what in a Logfire organization.

Reads the organization, its projects, members and roles, pending invitations, per-project
write and read tokens, organization API keys and Gateway providers through the Logfire
Platform API, then prints a report with findings a security reviewer asks about: credentials
that never expire, too many admins, guests, projects nobody can write to, expired tokens that
were never revoked.

Authentication is an organization API key (`LOGFIRE_API_KEY`) created under
Organization -> Settings -> API Keys. The scopes it needs are listed in `ENDPOINTS`; an
endpoint the key cannot read is reported as such rather than failing the whole audit.

The audit is also emitted to Logfire as a single log line with the counts as attributes, so
an alert can watch for `admin_count > 3` or `credentials_without_expiry > 0`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
import logfire

from demo_core import configure_logfire

DEFAULT_BASE_URL = 'https://api-us.pydantic.dev'

# path -> scope the Platform API requires (from its OpenAPI spec).
ENDPOINTS: dict[str, str] = {
    '/v1/organization/': 'organization:read',
    '/v1/projects/': 'project:read',
    '/v1/members/': 'organization:read_member',
    '/v1/invitations/': 'organization:read_invitation',
    '/v1/api-keys/': 'organization:read_api_key',
    '/v1/gateway/providers/': 'organization:read',
    '/v1/projects/{project_id}/write-tokens/': 'project:write_token',
    '/v1/projects/{project_id}/read-tokens/': 'project:read_token',
}
REQUIRED_SCOPES = sorted(set(ENDPOINTS.values()))


@dataclass
class Fetch:
    """One endpoint read: the data, or why it could not be read."""

    path: str
    ok: bool
    data: Any = None
    status: int | None = None
    detail: str | None = None
    scope: str | None = None


@dataclass
class Finding:
    severity: str  # 'high' | 'medium' | 'info'
    message: str


@dataclass
class Report:
    base_url: str
    organization: dict[str, Any] | None = None
    projects: list[dict[str, Any]] = field(default_factory=list)
    members: list[dict[str, Any]] = field(default_factory=list)
    invitations: list[dict[str, Any]] = field(default_factory=list)
    api_keys: list[dict[str, Any]] = field(default_factory=list)
    gateway_providers: list[dict[str, Any]] = field(default_factory=list)
    write_tokens: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # project name -> tokens
    read_tokens: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    unreadable: list[Fetch] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def admins(self) -> list[dict[str, Any]]:
        return [m for m in self.members if m.get('role', '').lower() == 'admin']

    @property
    def guests(self) -> list[dict[str, Any]]:
        return [m for m in self.members if m.get('role', '').lower() == 'guest']


class PlatformClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, *, transport: httpx.BaseTransport | None = None):
        self._client = httpx.Client(
            base_url=base_url.rstrip('/') + '/api',
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=30,
            transport=transport,
        )

    def get(self, path: str, **params: Any) -> Fetch:
        scope = ENDPOINTS.get(_template(path))
        try:
            response = self._client.get(path, params=params or None)
        except httpx.HTTPError as exc:
            return Fetch(path, ok=False, detail=f'{type(exc).__name__}: {exc}', scope=scope)
        if response.status_code == 200:
            return Fetch(path, ok=True, data=response.json(), status=200, scope=scope)
        try:
            detail = response.json().get('detail')
        except ValueError:
            detail = response.text[:200]
        return Fetch(path, ok=False, status=response.status_code, detail=str(detail), scope=scope)

    def close(self) -> None:
        self._client.close()


def _template(path: str) -> str:
    """Map a concrete `/v1/projects/<uuid>/write-tokens/` back to its templated key."""
    parts = path.strip('/').split('/')
    if len(parts) == 4 and parts[1] == 'projects':
        return f'/v1/projects/{{project_id}}/{parts[3]}/'
    return path


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def collect(client: PlatformClient, *, max_admins: int) -> Report:
    report = Report(base_url=str(client._client.base_url).rstrip('/').removesuffix('/api'))

    def read(path: str, **params: Any) -> Any:
        fetch = client.get(path, **params)
        if not fetch.ok:
            report.unreadable.append(fetch)
            return None
        return fetch.data

    report.organization = read('/v1/organization/')
    report.projects = read('/v1/projects/') or []
    report.members = read('/v1/members/') or []
    report.invitations = read('/v1/invitations/') or []
    report.api_keys = read('/v1/api-keys/') or []
    providers = read('/v1/gateway/providers/', limit=100)
    report.gateway_providers = (providers or {}).get('providers', []) if isinstance(providers, dict) else []

    for project in report.projects:
        pid, name = project['id'], project['project_name']
        write = read(f'/v1/projects/{pid}/write-tokens/')
        if write is not None:
            report.write_tokens[name] = write
        read_ = read(f'/v1/projects/{pid}/read-tokens/')
        if read_ is not None:
            report.read_tokens[name] = read_

    report.findings = analyse(report, max_admins=max_admins)
    return report


def analyse(report: Report, *, max_admins: int) -> list[Finding]:
    findings: list[Finding] = []
    now = datetime.now(UTC)

    if len(report.admins) > max_admins:
        names = ', '.join(m.get('email', m.get('name', '?')) for m in report.admins)
        findings.append(Finding('medium', f'{len(report.admins)} organization admins (limit {max_admins}): {names}'))
    if report.guests:
        findings.append(
            Finding('info', f'{len(report.guests)} guest(s) with direct project access; confirm each is still needed')
        )

    for invitation in report.invitations:
        if invitation.get('max_usage_count', 1) > 1 or not invitation.get('expiration'):
            findings.append(Finding('medium', f'invitation {invitation["id"]} is reusable or never expires'))

    def check_credentials(kind: str, owner: str, items: list[dict[str, Any]]) -> None:
        for item in items:
            label = f'{kind} {item.get("name") or item.get("description") or item.get("token_prefix") or item["id"]} ({owner})'
            expires = _parse_dt(item.get('expires_at'))
            if item.get('active', True) is False:
                continue
            if expires is None:
                findings.append(Finding('high', f'{label} never expires'))
            elif expires < now:
                findings.append(
                    Finding('high', f'{label} expired on {expires:%Y-%m-%d} but is still active; revoke it')
                )
            if kind == 'api key' and item.get('all_projects'):
                findings.append(Finding('info', f'{label} is scoped to all projects'))

    for project, tokens in report.write_tokens.items():
        check_credentials('write token', project, tokens)
        if not [t for t in tokens if t.get('active', True)]:
            findings.append(
                Finding('info', f'project {project} has no active write token (nothing can send data to it)')
            )
    for project, tokens in report.read_tokens.items():
        check_credentials('read token', project, tokens)
    check_credentials(
        'api key',
        report.organization.get('organization_name', 'org') if report.organization else 'org',
        report.api_keys,
    )

    for fetch in report.unreadable:
        findings.append(
            Finding(
                'info',
                f'could not read {fetch.path}: HTTP {fetch.status or "-"} {fetch.detail}; needs scope {fetch.scope}',
            )
        )
    return findings


def render(report: Report) -> str:
    lines: list[str] = []
    org = report.organization or {}
    lines.append(f'Logfire access audit  ({report.base_url})')
    lines.append('=' * 72)
    if org:
        lines.append(
            f'Organization: {org.get("organization_display_name") or org.get("organization_name")}  '
            f'plan={org.get("subscription_plan")}  gateway_enabled={org.get("gateway_enabled")}'
        )
    lines.append(f'Projects ({len(report.projects)}):')
    for p in report.projects:
        w = len([t for t in report.write_tokens.get(p['project_name'], []) if t.get('active', True)])
        r = len([t for t in report.read_tokens.get(p['project_name'], []) if t.get('active', True)])
        lines.append(
            f'  - {p["project_name"]:<32} visibility={p.get("visibility"):<8} write_tokens={w} read_tokens={r}'
        )
    lines.append(f'Members ({len(report.members)}): {len(report.admins)} admin, {len(report.guests)} guest')
    for m in sorted(report.members, key=lambda m: (m.get('role', ''), m.get('email', ''))):
        lines.append(f'  - {m.get("email", "?"):<40} {m.get("role", "?")}')
    lines.append(f'Pending invitations: {len(report.invitations)}')
    lines.append(f'Organization API keys ({len(report.api_keys)}):')
    for k in report.api_keys:
        scope = 'all projects' if k.get('all_projects') else (k.get('project_name') or 'org')
        lines.append(
            f'  - {k.get("name", "?"):<32} scope={scope:<20} expires={k.get("expires_at") or "never":<26} '
            f'last_used={k.get("last_used_at") or "-"}  scopes={len(k.get("scopes", []))}'
        )
    for kind, per_project in (('Write tokens', report.write_tokens), ('Read tokens', report.read_tokens)):
        lines.append(f'{kind}:')
        for project, tokens in per_project.items():
            for t in tokens:
                lines.append(
                    f'  - {project:<28} {t.get("token_prefix", "?"):<14} active={t.get("active")} '
                    f'created={str(t.get("created_at", ""))[:10]} expires={str(t.get("expires_at") or "never")[:10]} '
                    f'by={t.get("created_by_name") or "-"}'
                )
    lines.append(
        f'Gateway providers: {", ".join(p.get("slug", "?") for p in report.gateway_providers) or "none / not readable"}'
    )
    lines.append('')
    lines.append(f'Findings ({len(report.findings)}):')
    for f in report.findings:
        lines.append(f'  [{f.severity:<6}] {f.message}')
    if not report.findings:
        lines.append('  none')
    return '\n'.join(lines)


def emit(report: Report) -> None:
    """One log line with counts as attributes: the thing an alert query can watch."""
    org = (report.organization or {}).get('organization_name', 'unknown')
    logfire.info(
        'Logfire access audit for {organization}',
        organization=org,
        project_count=len(report.projects),
        member_count=len(report.members),
        admin_count=len(report.admins),
        guest_count=len(report.guests),
        pending_invitations=len(report.invitations),
        api_key_count=len(report.api_keys),
        write_token_count=sum(len(v) for v in report.write_tokens.values()),
        read_token_count=sum(len(v) for v in report.read_tokens.values()),
        credentials_without_expiry=sum(1 for f in report.findings if 'never expires' in f.message),
        high_findings=sum(1 for f in report.findings if f.severity == 'high'),
        unreadable_endpoints=[f.path for f in report.unreadable],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Audit roles, tokens and API keys in a Logfire organization.')
    parser.add_argument('--json', action='store_true', help='print the report as JSON instead of text')
    parser.add_argument('--max-admins', type=int, default=3, help='flag when more org admins than this (default 3)')
    parser.add_argument('--base-url', default=os.environ.get('LOGFIRE_API_BASE_URL', DEFAULT_BASE_URL))
    parser.add_argument('--no-logfire', action='store_true', help='do not emit the audit summary to Logfire')
    args = parser.parse_args(argv)

    configure_logfire('rbac', instrument_pydantic_ai=False, console=False)
    api_key = os.environ.get('LOGFIRE_API_KEY')
    if not api_key:
        print(
            'LOGFIRE_API_KEY is not set. Create an organization API key in Logfire under '
            'Organization -> Settings -> API Keys with these scopes and put it in .env:\n  '
            + ', '.join(REQUIRED_SCOPES),
            file=sys.stderr,
        )
        return 2

    client = PlatformClient(api_key, args.base_url)
    try:
        with logfire.span('rbac audit'):
            report = collect(client, max_admins=args.max_admins)
            if not args.no_logfire:
                emit(report)
    finally:
        client.close()

    if args.json:
        print(json.dumps(asdict(report), indent=2, default=str))
    else:
        print(render(report))
    logfire.force_flush()
    return 1 if any(f.severity == 'high' for f in report.findings) else 0


if __name__ == '__main__':
    sys.exit(main())
