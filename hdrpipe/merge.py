"""Fusion de las tres tomas en un unico mapa de radiancia.

Esto es lo que en Lightroom haces con "Combinar > HDR", pero sin el tone mapping
que Lightroom aplica despues. El resultado es una imagen lineal sin techo: el
sofa vale 0.03 y la ventana puede valer 60. Toda la informacion de la ventana
sigue ahi, y por eso el window pull posterior no se inventa nada, simplemente
baja lo que ya se habia medido.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from .imageops import EPS, smoothstep, srgb_decode, srgb_encode, to_uint8
from .raw import Frame

log = logging.getLogger(__name__)


@dataclass
class MergeResult:
    image: np.ndarray               # float32 (H, W, 3), radiancia lineal
    method: str                     # "radiance" | "fusion" | "single"
    exposure_scales: list[float] = field(default_factory=list)
    ev_spread: float = 0.0          # paradas entre la toma mas clara y la mas oscura
    unrecoverable: float = 0.0      # fraccion de pixeles quemados en TODAS las tomas
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Escalas de exposicion
# --------------------------------------------------------------------------


def estimate_scales_from_pixels(frames: list[Frame]) -> list[float]:
    """Deduce las exposiciones relativas comparando los pixeles bien expuestos.

    Red de seguridad para cuando el EXIF no trae tiempo de exposicion o ISO
    (ocurre con algunos DNG procesados por apps de terceros).
    """
    reference = frames[0].image
    scales = [1.0]
    for frame in frames[1:]:
        a = reference.reshape(-1)
        b = frame.image.reshape(-1)
        # Solo pixeles con senal util en ambas tomas: ni ruido ni quemados.
        valid = (a > 0.02) & (a < 0.9) & (b > 0.02) & (b < 0.9)
        if valid.sum() < 1000:
            scales.append(1.0)
            continue
        scales.append(float(np.median(b[valid] / a[valid])))
    return scales


def resolve_exposure_scales(frames: list[Frame]) -> tuple[list[float], list[str]]:
    """Escala de exposicion de cada toma, del EXIF si se puede y si no, de los pixeles."""
    warnings: list[str] = []
    scales = [frame.exposure_scale for frame in frames]

    if all(scale is not None and scale > 0 for scale in scales):
        return [float(scale) for scale in scales], warnings  # type: ignore[arg-type]

    warnings.append("EXIF de exposicion incompleto; se estiman las escalas desde los pixeles")
    return estimate_scales_from_pixels(frames), warnings


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def _weights(image: np.ndarray, low_cut: float, high_cut: float) -> np.ndarray:
    """Peso por pixel: cuanto nos fiamos de esta toma en este punto.

    El peso se calcula sobre el canal mas alto, no sobre la luminancia: si el
    rojo esta quemado el color del pixel ya es falso aunque el verde este bien.
    """
    peak = image.max(axis=2)

    # Se descarta lo que esta en el ruido y lo que esta quemado, con transiciones
    # suaves para que no aparezcan costuras entre zonas de distinta procedencia.
    usable = smoothstep(peak, low_cut, low_cut * 5.0 + EPS) * (
        1.0 - smoothstep(peak, high_cut * 0.82, high_cut)
    )

    # Dentro de lo usable, se prefiere lo que cae en la zona media de la curva,
    # donde el sensor tiene mejor relacion senal/ruido.
    centered = np.exp(-((np.power(np.clip(peak, 0, 1), 1 / 2.2) - 0.5) ** 2) / (2 * 0.28**2))

    return (usable * centered).astype(np.float32)


def merge_radiance(
    frames: list[Frame],
    *,
    low_cut: float = 0.005,
    high_cut: float = 0.985,
) -> MergeResult:
    """Fusion fisica: media ponderada de las tomas llevadas a escala comun."""
    scales, warnings = resolve_exposure_scales(frames)

    order = np.argsort(scales)                 # de la mas oscura a la mas clara
    darkest, brightest = frames[order[0]], frames[order[-1]]
    darkest_scale, brightest_scale = scales[order[0]], scales[order[-1]]

    accumulator = np.zeros_like(frames[0].image, dtype=np.float32)
    weight_sum = np.zeros(frames[0].image.shape[:2], dtype=np.float32)

    for frame, scale in zip(frames, scales):
        weight = _weights(frame.image, low_cut, high_cut)
        accumulator += (frame.image / max(scale, EPS)) * weight[..., None]
        weight_sum += weight

    unweighted = weight_sum < 1e-4

    radiance = np.zeros_like(accumulator)
    np.divide(accumulator, np.maximum(weight_sum, EPS)[..., None], out=radiance)

    # Pixeles sin ninguna toma fiable: o estan quemados hasta en la toma mas
    # oscura, o son negro puro hasta en la mas larga. Se toma la mejor
    # aproximacion disponible en lugar de dejar un agujero.
    if unweighted.any():
        bright_side = darkest.image.max(axis=2) > 0.5
        fallback = np.where(
            bright_side[..., None],
            darkest.image / max(darkest_scale, EPS),
            brightest.image / max(brightest_scale, EPS),
        )
        radiance[unweighted] = fallback[unweighted]

    # Se renormaliza a la escala de la toma central: la imagen resultante se
    # parece a esa toma, pero con las altas luces por encima de 1.0 intactas.
    middle_scale = float(np.median(scales))
    radiance *= middle_scale

    ev_spread = float(np.log2(max(brightest_scale, EPS) / max(darkest_scale, EPS)))
    burnt = float((darkest.image.max(axis=2) >= high_cut).mean())

    return MergeResult(
        image=np.ascontiguousarray(radiance, dtype=np.float32),
        method="radiance",
        exposure_scales=[float(s) for s in scales],
        ev_spread=ev_spread,
        unrecoverable=burnt,
        warnings=warnings,
    )


def merge_fusion(frames: list[Frame]) -> MergeResult:
    """Exposure fusion de Mertens: no necesita EXIF ni escalas.

    Ultimo recurso. Produce una imagen ya en rango de pantalla, asi que se
    devuelve linealizada para que el resto del pipeline no tenga que saberlo,
    pero no hay altas luces por encima de 1.0 que recuperar.
    """
    ldr = [to_uint8(srgb_encode(np.clip(frame.image, 0.0, 1.0))) for frame in frames]
    merged = cv2.createMergeMertens().process([img.astype(np.float32) / 255.0 for img in ldr])
    merged = np.clip(merged, 0.0, 1.0).astype(np.float32)
    return MergeResult(
        image=srgb_decode(merged),
        method="fusion",
        warnings=["fusion Mertens: sin datos de exposicion, el window pull sera limitado"],
    )


def merge_frames(
    frames: list[Frame],
    *,
    mode: str = "radiance",
    low_cut: float = 0.005,
    high_cut: float = 0.985,
) -> MergeResult:
    """Punto de entrada del modulo."""
    if not frames:
        raise ValueError("no hay tomas que fusionar")
    if len(frames) == 1:
        return MergeResult(
            image=frames[0].image.copy(),
            method="single",
            warnings=["una sola toma: sin rango dinamico extra que aprovechar"],
        )

    if mode == "fusion":
        return merge_fusion(frames)

    result = merge_radiance(frames, low_cut=low_cut, high_cut=high_cut)

    # Si las escalas salen practicamente identicas no habia bracket real y la
    # fusion no aporta nada; Mertens al menos saca contraste de las tres tomas.
    if result.ev_spread < 0.3 and len(frames) > 1:
        log.warning("separacion de exposicion casi nula (%.2f EV)", result.ev_spread)
        fallback = merge_fusion(frames)
        fallback.warnings = result.warnings + [
            f"separacion de exposicion de solo {result.ev_spread:.2f} EV; se usa fusion Mertens"
        ] + fallback.warnings
        return fallback

    return result
