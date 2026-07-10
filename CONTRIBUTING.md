# Contributing

Thank you for considering an improvement to Vardaan.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Before Opening a Pull Request

Run:

```bash
ruff check .
pytest
```

## Contribution Guidelines

- Keep medical claims conservative and cite reliable sources.
- Do not present model output as clinical advice.
- Never commit API keys, `.env` files, databases, logs, or personal data.
- Keep training and inference feature order consistent.
- Add or update tests when application behavior changes.
- Explain model changes and include evaluation results.
- Use clear commit messages such as:
  - `docs: improve model limitations section`
  - `fix: validate missing prediction fields`
  - `test: add health endpoint coverage`
  - `feat: add calibrated probability output`

## Reporting Issues

Include:

- What happened
- What you expected
- Steps to reproduce
- Python version
- Relevant error output with secrets removed
