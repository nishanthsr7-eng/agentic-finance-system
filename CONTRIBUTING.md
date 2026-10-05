# Contributing

Thanks for taking a look. Issues and pull requests are welcome.

## Setup

`docker compose up --build` runs the full stack. For a local Python setup, see
[docs/SETUP.md](docs/SETUP.md).

## Before opening a pull request

```bash
ruff check . && ruff format --check .
mypy backend
pytest backend/prediction -q
pytest backend/tests -q
npm run test:e2e
```

CI runs the same checks on every pull request.

## Conventions

- Branches: `<type>/<what-changes>`, e.g. `fix/marketplace-sell-cap`.
- Commits: [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `chore:`).
- Model changes: a new feature or model only ships if it passes its research
  gate out of sample (see [scripts/experiments/README.md](scripts/experiments/README.md)).
  Report results against the always-up baseline.
