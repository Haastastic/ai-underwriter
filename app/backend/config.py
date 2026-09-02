"""Backend settings, resolved from the environment with sensible defaults.

Kept as a plain dataclass (not pydantic-settings) so the model layer and
tests can build one explicitly without any framework magic.

`load_env_file()` reads a local `.env` (git-ignored) into the process
environment before settings and the LLM client are resolved. Real
environment variables always win over `.env`, and tests never call it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.llm.client import DEFAULT_MODEL
from src.model.decision import APPROVE_BELOW, DENY_AT_OR_ABOVE

ENV_FILE = Path(".env")

# Where the Phase 7 loan-officer frontend runs during development. The built
# static app served from the same origin as the API needs no entry here; a
# separately hosted frontend adds its origin via AIU_CORS_ORIGINS.
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def load_env_file(path: str | Path = ENV_FILE) -> bool:
    """Load key=value pairs from `path` into os.environ if the file exists.

    Existing environment variables are not overridden. Returns True if a
    file was found and read. Imported lazily so the dependency is only
    needed when this is actually called (the server entrypoint), not by the
    test suite.
    """
    path = Path(path)
    if not path.is_file():
        return False
    from dotenv import load_dotenv

    load_dotenv(path, override=False)
    return True


@dataclass(frozen=True)
class Settings:
    models_root: Path = Path("models")
    model_version: str = "v2"
    training_data_path: Path = Path("data/raw/cs-training.csv")
    db_path: Path = Path("underwriter.db")
    llm_model: str = DEFAULT_MODEL
    max_reasons: int = 4
    # Decision policy is config, not code: a new model version can be scored
    # with cutoffs tuned to its own calibration without a code change.
    approve_below: float = APPROVE_BELOW
    deny_at_or_above: float = DENY_AT_OR_ABOVE
    # Browser origins allowed to call the API (the frontend dev server).
    cors_origins: tuple[str, ...] = field(default_factory=lambda: DEFAULT_CORS_ORIGINS)

    @property
    def model_dir(self) -> Path:
        return self.models_root / self.model_version


def settings_from_env() -> Settings:
    """Build Settings, letting AIU_* environment variables override defaults."""
    base = Settings()
    return Settings(
        models_root=Path(os.environ.get("AIU_MODELS_ROOT", base.models_root)),
        model_version=os.environ.get("AIU_MODEL_VERSION", base.model_version),
        training_data_path=Path(
            os.environ.get("AIU_TRAINING_DATA", base.training_data_path)
        ),
        db_path=Path(os.environ.get("AIU_DB_PATH", base.db_path)),
        llm_model=os.environ.get("AIU_LLM_MODEL", base.llm_model),
        max_reasons=int(os.environ.get("AIU_MAX_REASONS", base.max_reasons)),
        approve_below=float(
            os.environ.get("AIU_APPROVE_BELOW", base.approve_below)
        ),
        deny_at_or_above=float(
            os.environ.get("AIU_DENY_AT_OR_ABOVE", base.deny_at_or_above)
        ),
        cors_origins=_split_origins(
            os.environ.get("AIU_CORS_ORIGINS"), base.cors_origins
        ),
    )


def _split_origins(raw: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a comma-separated origin list; blank/unset falls back to default."""
    if not raw or not raw.strip():
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())
