# Contributing

Thanks for taking an interest. Bug reports, feedback from real setups, and
patches are all welcome.

- **Setting up a dev environment, running the offline test suite, and the
  code conventions** are covered in [docs/development.md](docs/development.md).
- **Bugs and feature requests** go through
  [issues](https://github.com/Dixiao-L/gustarr/issues); the templates ask
  for the few details that make problems reproducible (versions, config
  shape with secrets removed, relevant log lines).
- **Pull requests**: keep the suite green (`uv run pytest`), keep
  `uv run ruff check .` clean, and add a changelog entry for anything a
  user would notice. Tests are offline by design — anything that would
  touch the network in tests is a bug.
- **Security issues**: please use
  [private vulnerability reporting](https://github.com/Dixiao-L/gustarr/security/advisories/new)
  instead of a public issue — see [SECURITY.md](SECURITY.md).
