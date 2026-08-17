"""Carga y fusion de la configuracion del pipeline.

La configuracion vive en YAML y se accede con puntos (`cfg.tone.window_pull.ev`)
para que el codigo del pipeline se lea como la receta que es.

Orden de precedencia, de menor a mayor:
    config/default.yaml  ->  config/learned.yaml  ->  overrides en memoria
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"
LEARNED_CONFIG_PATH = CONFIG_DIR / "learned.yaml"


class Section(Mapping[str, Any]):
    """Diccionario de solo lectura con acceso por atributo."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]):
        wrapped: dict[str, Any] = {}
        for key, value in data.items():
            wrapped[key] = Section(value) if isinstance(value, Mapping) else value
        object.__setattr__(self, "_data", wrapped)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(
                f"la configuracion no tiene la clave '{name}' "
                f"(disponibles: {', '.join(sorted(self._data))})"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("la configuracion es de solo lectura; usa cfg.with_overrides()")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value.to_dict() if isinstance(value, Section) else copy.deepcopy(value)
            for key, value in self._data.items()
        }

    def __repr__(self) -> str:  # pragma: no cover - ayuda de depuracion
        return f"Section({sorted(self._data)})"


class Config(Section):
    """Configuracion completa del pipeline."""

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        """Devuelve una copia nueva con `overrides` fusionado encima."""
        return Config(deep_merge(self.to_dict(), overrides))

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Fusiona `overrides` sobre `base` recursivamente, sin mutar ninguno."""
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} debe contener un diccionario YAML en la raiz")
    return dict(data)


def load_config(
    *,
    path: Path | None = None,
    learned: Path | None = LEARNED_CONFIG_PATH,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Carga la configuracion aplicando defaults, valores calibrados y overrides."""
    data = _read_yaml(path or DEFAULT_CONFIG_PATH)
    if learned is not None:
        data = deep_merge(data, _read_yaml(learned))
    if overrides:
        data = deep_merge(data, overrides)
    return Config(data)
