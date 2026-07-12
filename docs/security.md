# Security Model

Gustarr deliberately contains **no authentication code** — no passwords, no
sessions, no tokens. It *maps* identities that the deployment vouches for;
it never establishes them. This is a design decision, not an omission: a
household recommender re-implementing login would duplicate (worse) what a
reverse proxy, SSO stack, or network boundary already does well, and every
line of auth code is a line that can be wrong.

That decision only works if the trust boundaries are explicit. They are:

## The layers

| Layer | What it authenticates | Provided by |
|---|---|---|
| Network reachability | Who can talk to the port at all | Loopback bind (default), firewall rules, Docker network isolation |
| Reverse proxy access | Who may reach the app path | Your proxy's access control (IP allowlists, VPN) |
| Forward-auth / SSO | *Which person* is making the request | Authelia/Authentik/oauth2-proxy → sets the `Remote-User` header (name configurable via `[web] profile_header`) |
| Gustarr profile mapping | Which taste model the request acts on | `get_profile`: header → `?profile=` → sole profile → `default`; unknown names are rejected (403) |
| In-app request guards | Browser-borne attacks, not people | Host allowlist (DNS rebinding) + Origin check on mutating methods (cross-site POSTs) |

Two properties of the mapping worth knowing:

- **The auth header always beats `?profile=`.** Behind a forward-auth proxy
  you cannot impersonate another profile by editing the URL — your proxy
  identity wins.
- **`?profile=` exists for the *unauthenticated* paths** — CLI-adjacent use,
  SSH tunnels, single-operator setups. On those paths there is no identity
  to contradict.

## What this means in practice

- **Single profile (the default):** nothing to decide. Whoever can reach the
  port operates the queue; keep the port loopback/intranet-only as shipped.
- **Multiple profiles behind an authenticating proxy** (the intended
  multi-user shape): identity is as strong as your SSO. Point
  `profile_header` at the header your proxy sets, and make sure the proxy is
  the *only* network path to the port — a client that can reach Gustarr
  directly can write any header it likes.
- **Multiple profiles without an authenticating proxy:** profile identity is
  cosmetic — any household member can act as any other. Gustarr logs an
  advisory at startup in this shape. That may be perfectly fine for your
  living room; it is not access control, and Gustarr won't pretend otherwise.

## Authorization

- Approve/reject/snooze/forgive act on **the resolved profile's** queue and
  taste model only.
- **Runtime settings are operator-level and global** — any person your
  deployment lets in can pause automation or change budgets for everyone.
  There is no per-profile settings authority; a household that needs that
  boundary should keep the Settings dialog behind a stricter proxy rule
  (e.g. Authelia group policy on `PUT/DELETE /api/settings`).
- The pipeline and CLI run with filesystem access to the store; anyone with
  shell access to the host is an operator by definition.

## Secrets

API keys live in the process environment (`env:VAR` in the TOML), never in
the store, never in the web UI, never in API responses. The store contains
taste data (events, embeddings, verdicts) — treat backups accordingly.
