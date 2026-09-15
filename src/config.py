"""
Central config loader. Every layer imports `get_settings()` instead of
reading YAML directly, so config parsing happens once and stays in one place.
"""
from __future__ import annotations

import os
import logging
import logging.config
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)


@lru_cache(maxsize=1)
def get_settings() -> dict:
    config_path = ROOT_DIR / "configs" / "settings.yaml"
    with open(config_path, "r") as f:
        settings = yaml.safe_load(f)

    # env overrides for anything secret / deployment-specific
    settings["llm"]["provider"] = os.getenv("LLM_PROVIDER", settings["llm"]["provider"])
    settings["llm"]["model"] = os.getenv("LLM_MODEL", settings["llm"]["model"])
    settings["embeddings"]["model"] = os.getenv(
        "EMBEDDING_MODEL", settings["embeddings"]["model"]
    )
    return settings


def setup_logging() -> None:
    log_config_path = ROOT_DIR / "configs" / "logging.yaml"
    with open(log_config_path, "r") as f:
        log_config = yaml.safe_load(f)
    logging.config.dictConfig(log_config)


def get_path(key: str) -> Path:
    """Resolve a path from settings['paths'] relative to project root."""
    settings = get_settings()
    rel = settings["paths"][key]
    return (ROOT_DIR / rel).resolve()
