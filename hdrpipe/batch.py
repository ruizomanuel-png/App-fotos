"""Consistencia dentro del lote.

Treinta fotos de la misma casa editadas cada una por su cuenta salen bien una a
una y mal en conjunto: el salon algo mas calido que la cocina, un dormitorio
medio punto mas oscuro. Al pasarlas seguidas en un anuncio, eso canta.

La correccion se calcula en una primera pasada a baja resolucion (rapida) y se
aplica en la pasada definitiva, antes de codificar el JPEG. Asi no hay que
guardar en memoria cuarenta imagenes de 40 megapixeles ni recomprimir dos veces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .imageops import EPS, srgb_decode, srgb_encode

# Campos de la firma que se armonizan y como se combinan.
_TONE_KEYS = ("shadow", "midtone", "highlight")
_COLOR_KEYS = ("r_over_g", "b_over_g")


@dataclass
class BatchProfile:
    """Consenso del lote: hacia donde se acerca cada foto."""

    target: dict[str, float] = field(default_factory=dict)
    consistency: float = 0.0
    sample_size: int = 0

    @property
    def active(self) -> bool:
        return self.sample_size >= 2 and self.consistency > 0.0


def build_profile(signatures: list[dict[str, float]], consistency: float) -> BatchProfile:
    """Consenso robusto del lote.

    Se usa la mediana y no la media porque un lote normal trae una o dos fotos
    atipicas (un bano diminuto, una toma a contraluz) que desplazarian la media.
    """
    if not signatures:
        return BatchProfile(consistency=float(consistency))

    target = {
        key: float(np.median([s[key] for s in signatures if key in s]))
        for key in (*_TONE_KEYS, *_COLOR_KEYS)
        if any(key in s for s in signatures)
    }
    return BatchProfile(
        target=target,
        consistency=float(np.clip(consistency, 0.0, 1.0)),
        sample_size=len(signatures),
    )


@dataclass
class Correction:
    exposure: float = 1.0
    red: float = 1.0
    blue: float = 1.0

    @property
    def is_identity(self) -> bool:
        return (
            abs(self.exposure - 1.0) < 1e-3
            and abs(self.red - 1.0) < 1e-3
            and abs(self.blue - 1.0) < 1e-3
        )


def correction_for(signature: dict[str, float], profile: BatchProfile) -> Correction:
    """Ganancias que acercan una foto al consenso, sin llegar a igualarla.

    Con `consistency` a 1.0 todas las fotos quedarian identicas en tono, lo que
    tambien esta mal: un bano interior debe verse mas recogido que un salon con
    tres ventanas. El valor por defecto corrige la deriva sin borrar la escena.
    """
    if not profile.active:
        return Correction()

    blend = profile.consistency

    def pull(key: str) -> float:
        own = signature.get(key)
        goal = profile.target.get(key)
        if not own or not goal or own <= EPS:
            return 1.0
        return float((goal / own) ** blend)

    # El tono se corrige sobre los medios, que es donde se percibe el brillo
    # general; sombras y altas luces se dejan a la escena.
    exposure_display = pull("midtone")
    # La ganancia se ha medido en espacio de pantalla; en lineal el efecto de
    # una misma proporcion es mas fuerte, asi que se atenua con la gamma.
    exposure = float(np.clip(exposure_display**2.2, 0.6, 1.6))

    return Correction(
        exposure=exposure,
        red=float(np.clip(pull("r_over_g"), 0.85, 1.18)),
        blue=float(np.clip(pull("b_over_g"), 0.85, 1.18)),
    )


def apply_correction(display: np.ndarray, correction: Correction) -> np.ndarray:
    """Aplica la correccion del lote en espacio lineal.

    En lineal y no sobre los valores con gamma: una ganancia aplicada
    directamente sobre valores codificados desplaza los medios de forma
    distinta que las sombras y ensucia el color.
    """
    if correction.is_identity:
        return display

    linear = srgb_decode(display)
    gains = np.array(
        [correction.exposure * correction.red, correction.exposure, correction.exposure * correction.blue],
        dtype=np.float32,
    )
    return srgb_encode(np.clip(linear * gains, 0.0, 1.0))
