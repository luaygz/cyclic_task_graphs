"""Configuration files reject unknown keys, so a typo cannot silently fall back to a default."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from ctg.config import CONFIGS_DIR, ExperimentConfig, ModelAssignment, load_model_presets, resolve_models

BASE: dict[str, Any] = {"name": "t", "benchmark": "textcraft", "textcraft_depth": 2, "controller": "spec_cyc"}
NSHOT_CONFIGS = sorted((CONFIGS_DIR / "nshot").glob("*.yaml"))


def test_shipped_presets_load_and_cover_the_default_assignment() -> None:
    resolve_models(ModelAssignment(), load_model_presets())


@pytest.mark.parametrize("path", NSHOT_CONFIGS, ids=lambda path: path.stem)
def test_shipped_nshot_configs_load_and_resolve(path: Path) -> None:
    config = ExperimentConfig.load(path)

    resolve_models(config.models, load_model_presets())


@pytest.mark.parametrize(
    ("override", "typo"),
    [
        pytest.param({"models": {"excutor": "gpt-5-mini"}}, "excutor", id="models"),
        pytest.param(
            {"budgets": {"global_tool_calls": 30, "local_tool_calls": 5, "locl": 1}}, "locl", id="budgets"
        ),
    ],
)
def test_unknown_key_in_a_nested_section_is_rejected(override: dict[str, Any], typo: str) -> None:
    with pytest.raises(ValidationError, match=typo):
        ExperimentConfig.model_validate({**BASE, **override})


def test_unknown_field_in_a_model_preset_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    preset = {"name": "gpt-5-mini", "reasoning_efort": "low", "supports_temperature": False}
    path.write_text(yaml.safe_dump({"gpt-5-mini": preset}))

    with pytest.raises(ValidationError, match="reasoning_efort"):
        load_model_presets(path)
