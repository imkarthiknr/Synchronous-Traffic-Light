# Contributing

Thanks for taking a look. This is a personal portfolio project, but issues and pull requests are welcome.

## Getting started

Follow [DEVELOPMENT.md](DEVELOPMENT.md) to set up, then check everything passes:

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

## Pull requests

- Keep each pull request to one change and explain what it changes and why.
- Add tests for new behaviour. The environment is quick to run for a few simulated minutes
  (`JunctionEnv(horizon=300)`), so prefer real simulation over mocks.
- If a change affects results (environment, demand, rewards, training), rerun the evaluation
  and update `results/` and the README table in the same pull request, with the command used.
- Compare controllers on the same seeds, and never on the seeds used for model selection.
- Don't commit run outputs, detector weights or large videos. The one trained model in
  `models/` is the model behind the published results; replace it only together with `results/`.
- Add a line to the [CHANGELOG](CHANGELOG.md).

## Commit messages

Use the imperative mood with a short summary line, for example `Add pressure reward`.
