# rbac

## What it shows

Who can see and send what in a Logfire organization, and how to prove it. Security and
platform reviewers ask the same questions before a rollout: which roles exist, which
credentials can read data versus only send it, whether anything never expires, and how
many admins there are. `rbac-audit` answers them from the Platform API in one run and emits
the counts to Logfire so an alert can watch them. `rbac-tokens` demonstrates the scope
boundary between a write token and an API key.

## Run

```bash
# Token separation: needs only LOGFIRE_TOKEN (already in .env)
uv run rbac-tokens

# Access audit: needs an organization API key in .env as LOGFIRE_API_KEY
uv run rbac-audit                 # human-readable report; exit code 1 if any high finding
uv run rbac-audit --json          # machine-readable
uv run rbac-audit --max-admins 2  # tighten the admin threshold (default 3)
```

Create the key in Logfire under **Organization -> Settings -> API Keys -> New API Key**,
scoped to all projects, with exactly these scopes:

```
organization:read, organization:read_member, organization:read_invitation,
organization:read_api_key, project:read, project:read_token, project:write_token
```

(`project:read_token` / `project:write_token` are the scopes that *list* read and write
tokens; they also allow creating them, so give the key an expiry and revoke it after the
audit.) For the EU region set `LOGFIRE_API_BASE_URL=https://api-eu.pydantic.dev`.

Without the key the audit prints the scope list and exits with code 2. With a key that lacks
some scopes it reports each unreadable endpoint and the scope it needs, and audits the rest.

## What to look at in Logfire

- `rbac-tokens` leaves one `rbac token separation demo` span in the project, sent with the
  write token. The Platform API call with the same token returns
  `403 This token is missing the required scope: project:read.` and is printed, not traced.
- `rbac-audit` emits one log line, `Logfire access audit for <org>`, with `admin_count`,
  `guest_count`, `api_key_count`, `write_token_count`, `credentials_without_expiry`,
  `high_findings` and `unreadable_endpoints` as attributes. Run it on a schedule and alert on it:

```sql
SELECT start_timestamp, attributes->>'organization' AS org,
       (attributes->'admin_count')::int AS admins,
       (attributes->'credentials_without_expiry')::int AS no_expiry,
       (attributes->'high_findings')::int AS high
FROM records
WHERE message LIKE 'Logfire access audit for %'
  AND ((attributes->'admin_count')::int > 3 OR (attributes->'high_findings')::int > 0)
```

## Roles

Organization roles (every member has one):

| Role | Project access | Editable permissions |
| --- | --- | --- |
| Admin | all public and private projects, all settings | no |
| Member | public projects; permissions can be adjusted | yes |
| Guest | none by default; created when someone is invited directly to a project | no |

Project roles (assigned per project; override the organization role's project permissions):

| Role | What it can do |
| --- | --- |
| Admin | everything in the project |
| Write | a configurable set of edit permissions |
| Read | read only |

Custom roles and externally managed groups (SCIM, Microsoft Entra ID, SSO) are Enterprise
features. Projects are the unit of access control; environments are only a filter, so if
different people may see prod versus staging data, use separate projects.

## Credentials

| Credential | Created at | Can | Cannot |
| --- | --- | --- | --- |
| Write token | Project -> Settings -> Write tokens | send traces, logs and metrics to that project (`project:write_otlp`) | read anything, manage anything |
| Read token | Project -> Settings -> Read tokens | query that project's data (`project:read_otlp`), e.g. for dashboards and the Query API | send data, manage settings |
| Project API key | Project -> Settings -> API Keys | exactly the scopes selected: send telemetry, query, manage tokens, alerts, dashboards, variables | anything outside that project |
| Organization API key | Organization -> Settings -> API Keys | organization-level scopes (members, invitations, API keys, audit logs, billing) and optionally project scopes across all or one project | send telemetry unless scoped to one project with "Send telemetry" |
| Personal API key | either of the above, marked personal | the same, limited to the creating user's own permissions | outlive that user's membership (it is deleted with it) |
| Gateway API key | Gateway -> API Keys | call models through the Gateway; carries daily/weekly/monthly/total spending limits and an expiry | read or send Logfire telemetry |

Service accounts and deployed apps should hold a write token or a "Send telemetry" project
API key and nothing more; everything that manages Logfire should use an API key with the
minimum scopes and an expiry. `rbac-tokens` shows the boundary; `rbac-audit` finds the
credentials that break the rule.

## A recommended enterprise setup

1. One project per environment that has different readers (`app-prod`, `app-staging`), so
   project roles do the access control; use `environment=` inside a project only to
   separate data the same people may see.
2. One write token per deployed service, named after the service, with an expiry; rotate
   with the API (`POST /v1/projects/{id}/write-tokens/{token}/rotate/`) rather than
   creating new ones by hand.
3. Read tokens or `project:read_otlp` API keys for dashboards, notebooks and the Query API;
   never a personal login.
4. Members by default, Admins by exception; guests only for time-boxed external access.
5. `include_content=False` on Pydantic AI instrumentation (see `apps/gateway-routing`) where
   prompts may carry personal data, and Logfire scrubbing for everything else.
6. SSO and SCIM provisioning on Enterprise so joiners and leavers are handled by the
   identity provider; audit logs via the API for the security team.
7. Run `rbac-audit` on a schedule and alert on the counts.

## Pre-rollout checklist

- Owners: who owns the organization, each project and each environment.
- Roles: who is Admin, who is Member, who is a Guest and until when.
- Credentials: every token and key has an owner, a purpose and an expiry; none are shared.
- Gateway: keys per team or app with spending limits; per-member limits for individuals.
- Content policy: which projects may receive prompt and completion text.
- Audit: the audit runs, its output is reviewed, and an alert watches the counts.

## Docs

- Organizations, projects and roles: https://pydantic.dev/docs/logfire/guides/web-ui/organizations-and-projects/
- API keys and scopes: https://pydantic.dev/docs/logfire/reference/advanced/use-api-keys/
- Write tokens: https://pydantic.dev/docs/logfire/how-to-guides/create-write-tokens/
- Environments versus projects: https://pydantic.dev/docs/logfire/how-to-guides/environments/
- SSO: https://pydantic.dev/docs/logfire/how-to-guides/sso-setup/
- SCIM provisioning (Enterprise): https://pydantic.dev/docs/logfire/scim-provisioning/
- Audit logs API (Enterprise): https://pydantic.dev/docs/logfire/audit-logs-api/
- Platform API reference (Swagger): https://api-us.pydantic.dev/api/docs and https://api-eu.pydantic.dev/api/docs
- Gateway API keys and spending limits: https://pydantic.dev/docs/logfire/reference/advanced/gateway/#api-keys

## Files

- `src/rbac/audit.py` — Platform API client, report model, findings, text/JSON rendering, Logfire emission.
- `src/rbac/tokens.py` — sends a span with the write token, then shows the same token being refused by the Platform API.
- `tests/test_audit.py` — offline tests against fixtures shaped like the Platform API schemas, including 401/403 handling.
