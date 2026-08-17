"""Control de calidad: decide que fotos van a la entrega y cuales a revisar.

Una foto que no pasa el control no se tira: se guarda igual, pero en la
subcarpeta `revisar/`, fuera del ZIP principal. Descartarla en silencio seria
perder una toma sin que nadie se entere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detail import sharpness_score
from .imageops import luminance


@dataclass
class QualityReport:
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def fail(self, reason: str) -> None:
        self.passed = False
        self.issues.append(reason)


def inspect(
    display: np.ndarray,
    cfg,
    *,
    bracket_size: int,
    blocking: list[str] | None = None,
) -> QualityReport:
    """Evalua el resultado final y decide si es entregable."""
    quality_cfg = cfg.quality
    report = QualityReport()

    if bracket_size < int(quality_cfg.min_brackets):
        report.fail(
            f"el bracket tenia {bracket_size} tomas y se esperaban "
            f"{quality_cfg.min_brackets}"
        )

    sharpness = sharpness_score(display)
    report.metrics["sharpness"] = sharpness
    if sharpness < float(quality_cfg.blur_threshold):
        report.fail(f"la imagen sale desenfocada o trepidada (nitidez {sharpness:.0f})")

    peak = display.max(axis=2)
    clipped_high = float((peak >= 0.996).mean())
    report.metrics["clipped_highlights"] = clipped_high
    if clipped_high > float(quality_cfg.max_clipped_highlights):
        report.fail(
            f"queda un {clipped_high:.1%} de altas luces quemadas tras la edicion"
        )

    lum = luminance(display)
    clipped_low = float((lum <= 0.004).mean())
    report.metrics["clipped_shadows"] = clipped_low
    if clipped_low > float(quality_cfg.max_clipped_shadows):
        report.fail(f"queda un {clipped_low:.1%} de sombras sin informacion")

    report.metrics["mean_luma"] = float(lum.mean())

    # Motivos de bloqueo que vienen de etapas anteriores (alineacion fallida,
    # objeto demasiado grande para borrar). El pipeline solo pasa aqui los
    # avisos que realmente comprometen la entrega, no los informativos.
    for reason in blocking or []:
        report.fail(reason)

    return report
