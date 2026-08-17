"""Agrupacion automatica de archivos sueltos en brackets.

De Drive llega una carpeta con 90 archivos y nadie los ha separado por escena.
Se agrupan por hora de disparo: tres tomas del mismo bracket estan a milisegundos
una de otra, y entre escena y escena pasan varios segundos porque hay que mover
el tripode.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .raw import SUPPORTED_EXTENSIONS, read_exif

log = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"(\d+)")


@dataclass
class Bracket:
    """Un grupo de tomas que corresponden a la misma escena."""

    name: str
    paths: list[Path]
    timestamp: datetime | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.paths)


def natural_key(path: Path) -> tuple:
    """Orden natural: IMG_9 va antes que IMG_10, al contrario que el orden ASCII."""
    parts = _NUMBER_RE.split(path.name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def list_supported_files(root: Path) -> list[Path]:
    """Busca archivos procesables bajo `root`, ignorando basura del sistema."""
    files = [
        path
        for path in sorted(Path(root).rglob("*"), key=natural_key)
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.name.startswith("._")       # recursos AppleDouble
        and "__MACOSX" not in path.parts
    ]
    return files


def _scene_name(paths: Sequence[Path]) -> str:
    """Nombre de la escena, sin el indice de la toma dentro del bracket.

    Las tres tomas suelen llamarse `salon_1`, `salon_2`, `salon_3`. Quedarse con
    el nombre del primer archivo dejaria el JPEG final como `salon_1.jpg`, que
    sugiere que hay un `salon_2.jpg` en alguna parte. Se usa el prefijo comun.
    """
    stems = [path.stem for path in paths]
    if len(stems) == 1:
        return stems[0]

    shortest = min(len(stem) for stem in stems)
    common = 0
    while common < shortest and len({stem[common] for stem in stems}) == 1:
        common += 1

    name = stems[0][:common].rstrip("_-. ")
    # Si los nombres apenas comparten nada (DSC0417, DSC0418...), el prefijo
    # comun no identifica la escena y es mejor el nombre completo del primero.
    return name if len(name) >= 3 else stems[0]


def _chunk(items: Sequence, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def group_brackets(
    paths: Sequence[Path],
    *,
    bracket_size: int = 3,
    max_seconds_between_shots: float = 8.0,
) -> list[Bracket]:
    """Agrupa archivos en brackets de `bracket_size` tomas.

    Con hora de disparo se agrupa por proximidad temporal; sin ella se recurre
    al orden natural del nombre de archivo, que es como salen de la camara.
    """
    if not paths:
        return []

    metadata: list[tuple[Path, datetime | None, float | None]] = []
    for path in paths:
        exif = read_exif(path)
        metadata.append((path, exif.get("timestamp"), exif.get("exposure_time")))

    has_timestamps = sum(1 for _, stamp, _ in metadata if stamp is not None)
    exposure_by_path = {path: exposure for path, _, exposure in metadata}

    if has_timestamps < len(metadata):
        if has_timestamps:
            log.warning(
                "%d de %d archivos no tienen hora de disparo; se agrupa por nombre",
                len(metadata) - has_timestamps,
                len(metadata),
            )
        ordered = sorted(paths, key=natural_key)
        clusters = [list(chunk) for chunk in _chunk(ordered, bracket_size)]
        stamps: list[datetime | None] = [None] * len(clusters)
    else:
        metadata.sort(key=lambda item: (item[1], natural_key(item[0])))
        clusters = []
        stamps = []
        current: list[Path] = []
        current_stamp: datetime | None = None
        previous: datetime | None = None

        for path, stamp, _ in metadata:
            gap = (stamp - previous).total_seconds() if previous is not None else 0.0
            starts_new_scene = (
                not current
                or gap > max_seconds_between_shots
                or len(current) >= bracket_size
            )
            if starts_new_scene and current:
                clusters.append(current)
                stamps.append(current_stamp)
                current = []
            if not current:
                current_stamp = stamp
            current.append(path)
            previous = stamp

        if current:
            clusters.append(current)
            stamps.append(current_stamp)

    brackets: list[Bracket] = []
    for cluster, stamp in zip(clusters, stamps):
        bracket = Bracket(name=_scene_name(cluster), paths=cluster, timestamp=stamp)
        if bracket.size != bracket_size:
            bracket.warnings.append(
                f"el bracket tiene {bracket.size} tomas y se esperaban {bracket_size}"
            )
        exposures = [exposure_by_path.get(path) for path in cluster]
        known = [value for value in exposures if value]
        if len(known) == len(cluster) and len(set(known)) == 1:
            bracket.warnings.append(
                "las tomas comparten tiempo de exposicion: puede que no sea un bracket"
            )
        brackets.append(bracket)

    return brackets
