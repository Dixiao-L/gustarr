# Security

## Reporting a vulnerability

Please report suspected vulnerabilities privately through
[GitHub's private vulnerability reporting](https://github.com/Dixiao-L/gustarr/security/advisories/new)
rather than a public issue. You should hear back within a week.

## Supported versions

Only the latest release receives fixes. Gustarr is pre-1.0; upgrading is
expected to be routine (stores migrate automatically, and the changelog
calls out anything that isn't).

## What Gustarr trusts

Gustarr deliberately authenticates nothing itself — it is designed to sit
behind a reverse proxy that does. Before exposing the web UI, read the
[trust model](docs/security.md): what the profile header means, why the
process must not be reachable directly, and what an attacker with network
access to it could do.
