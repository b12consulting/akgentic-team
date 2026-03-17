# akgentic-team

[![CI](https://github.com/b12consulting/akgentic-team/actions/workflows/ci.yml/badge.svg)](https://github.com/b12consulting/akgentic-team/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/jltournay/708bb547b8679308d083be7beaf4448a/raw/coverage.json)](https://github.com/b12consulting/akgentic-team/actions/workflows/ci.yml)

**Status:** Alpha - 1.0.0-alpha.1

Team lifecycle management for the Akgentic platform. Create, resume, stop, and delete multi-agent teams with event-sourced persistence and crash recovery.

## Installation

```bash
# Within the Akgentic workspace (from root)
uv sync --all-packages

# With MongoDB support
uv sync --all-packages --extra mongo

# With CLI
uv sync --all-packages --extra cli
```

## Dependencies

**Required:**
- `akgentic` (akgentic-core) - Actor framework, orchestrator, messaging
- `pydantic>=2.0.0` - Data validation and serialization

**Optional:**
- `pymongo>=4.0.0` - MongoDB persistence backend (`[mongo]` extra)
- `typer>=0.9.0`, `rich>=13.0.0` - CLI interface (`[cli]` extra)

## Development

```bash
# Run tests
uv run pytest packages/akgentic-team/tests/

# Type checking (strict mode)
uv run mypy packages/akgentic-team/src/

# Linting
ruff check packages/akgentic-team/src/
```
