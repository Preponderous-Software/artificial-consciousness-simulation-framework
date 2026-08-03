# Security Policy

## Supported Versions

This is a research/experimental project with no versioned releases. Security
fixes are applied to `main` only.

## Reporting a Vulnerability

If you find a security issue (e.g. in the standalone web dashboard's
process-management endpoints, the Discord webhook sink, or credential/config
handling), please **do not open a public issue**. Instead, report it privately
via [GitHub's private vulnerability reporting](../../security/advisories/new)
for this repository, or by contacting the repository owner directly.

Please include:

- A description of the issue and its potential impact
- Steps to reproduce, if possible
- Any relevant logs or configuration (with secrets redacted)

We'll acknowledge reports as soon as possible and follow up once a fix is
available. Please give us a reasonable window to address the issue before any
public disclosure.

## Notes on this project's attack surface

- `scripts/web.py` binds to `127.0.0.1` by default; `--host 0.0.0.0` and
  `--allow-remote-spawn` are explicit opt-ins with no built-in auth — do not
  expose them on an untrusted network without adding your own auth/reverse
  proxy in front.
- Secrets (API keys, Discord webhook URLs) are read from environment variables
  via `${VAR}` substitution in config, never committed to the repo — see
  `.env.example`.
