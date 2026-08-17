"""Mapeo de tono: de radiancia lineal a una imagen que se puede mirar.

Aqui vive el nucleo de lo que haces a mano en Lightroom con las mascaras:
bajar las ventanas y levantar las sombras hasta que blancos y negros quedan
parejos. La diferencia es que la fuerza de cada ajuste no es fija, se calcula
por foto a partir de lo que realmente hay en la escena.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .imageops import (
    EPS,
    feather_radius,
    gaussian,
    luminance,
    robust_percentiles,
    smoothstep,
    soft_mask,
    srgb_decode,
    srgb_encode,
)


@dataclass
class ToneResult:
    image: np.ndarray                       # espacio de pantalla, [0, 1]
    window_mask: np.ndarray                 # 1.0 dentro de las ventanas
    shadow_mask: np.ndarray
    applied_window_ev: float = 0.0
    applied_shadow_ev: float = 0.0
    exposure_gain: float = 1.0
    stats: dict = field(default_factory=dict)


def _log_luminance(linear: np.ndarray) -> np.ndarray:
    """Luminancia en paradas. Los percentiles sobre escala log se comportan bien
    aunque la escena tenga 16 paradas de rango."""
    return np.log2(np.maximum(luminance(linear), 1e-4)).astype(np.float32)


def _normalized_guide(log_lum: np.ndarray) -> np.ndarray:
    lo, hi = robust_percentiles(log_lum, [1.0, 99.0])
    if hi - lo < 1e-3:
        hi = lo + 1e-3
    return np.clip((log_lum - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def interior_level(lum: np.ndarray, percentile: float = 55.0) -> float:
    """Nivel de luminancia representativo del interior de la escena.

    Se descarta antes el 8% mas brillante: si no, una ventana grande arrastra
    la medida hacia arriba y el salon acaba saliendo oscuro.
    """
    ceiling = robust_percentiles(lum, [92.0])[0]
    interior = lum[lum <= max(ceiling, EPS)]
    if interior.size < 1000:
        interior = lum.reshape(-1)
    return float(np.percentile(interior, percentile))


def auto_exposure(
    linear: np.ndarray,
    *,
    target_midtone: float = 0.46,
    midtone_percentile: float = 55.0,
) -> float:
    """Ganancia que deja el interior de la escena en la luminancia objetivo."""
    current = interior_level(luminance(linear), midtone_percentile)
    if current <= EPS:
        return 1.0
    target_linear = float(srgb_decode(np.array([target_midtone], dtype=np.float32))[0])
    # Se acota para que una foto muy fallada no se dispare seis paradas.
    return float(np.clip(target_linear / current, 0.05, 20.0))


def _required_ev(
    log_lum: np.ndarray,
    mask: np.ndarray,
    percentile: float,
    target_display: float,
    max_ev: float,
    *,
    direction: int,
) -> float:
    """Paradas necesarias para llevar la zona enmascarada a su tono objetivo.

    `direction` es -1 para bajar (ventanas) y +1 para subir (sombras). El valor
    del YAML actua como tope, no como cantidad fija: una ventana con el jardin
    apenas quemado recibe menos correccion que una a contraluz directo.
    """
    selected = log_lum[mask > 0.5]
    if selected.size < 200:
        return 0.0

    observed = float(np.percentile(selected, percentile))
    target_linear = float(srgb_decode(np.array([target_display], dtype=np.float32))[0])
    target_log = float(np.log2(max(target_linear, 1e-4)))

    needed = (observed - target_log) if direction < 0 else (target_log - observed)
    return float(np.clip(needed, 0.0, max_ev))


def map_tone(linear: np.ndarray, cfg) -> ToneResult:
    """Lineal sin techo -> imagen en espacio de pantalla."""
    tone_cfg = cfg.tone

    gain = auto_exposure(
        linear,
        target_midtone=tone_cfg.target_midtone,
        midtone_percentile=tone_cfg.midtone_percentile,
    )
    working = linear * gain

    log_lum = _log_luminance(working)
    guide = _normalized_guide(log_lum)

    # Ancla de la escena: el nivel del interior, en paradas. Todos los umbrales
    # se expresan como distancia a este punto, de modo que funcionan igual en un
    # dormitorio en penumbra y en una cocina blanca a mediodia.
    anchor = float(np.log2(max(interior_level(luminance(working)), 1e-4)))

    window_cfg = tone_cfg.window_pull
    shadow_cfg = tone_cfg.shadows

    empty = np.zeros(working.shape[:2], dtype=np.float32)
    window_mask = empty
    shadow_mask = empty
    window_ev = 0.0
    shadow_ev = 0.0

    gain_stops = np.zeros(working.shape[:2], dtype=np.float32)

    if window_cfg.enabled:
        low, high = window_cfg.stops_above_interior
        window_mask = soft_mask(
            log_lum, guide, anchor + float(low), anchor + float(high), window_cfg.feather
        )
        window_ev = _required_ev(
            log_lum, window_mask, 98.0, window_cfg.target, window_cfg.ev, direction=-1
        )
        gain_stops -= window_ev * window_mask

    if shadow_cfg.enabled:
        low, high = shadow_cfg.stops_below_interior
        # `invert` hace que la mascara valga 1 por debajo del umbral inferior.
        shadow_mask = soft_mask(
            log_lum,
            guide,
            anchor - float(high),
            anchor - float(low),
            shadow_cfg.feather,
            invert=True,
        )
        shadow_ev = _required_ev(
            log_lum, shadow_mask, 12.0, shadow_cfg.target, shadow_cfg.ev, direction=+1
        )
        gain_stops += shadow_ev * shadow_mask

    if window_ev or shadow_ev:
        working = working * np.exp2(gain_stops)[..., None]

    display = _filmic(working, white_point=tone_cfg.highlight_rolloff)

    if window_cfg.enabled and window_cfg.local_contrast > 0 and window_ev > 0:
        display = _local_contrast(display, window_mask, window_cfg.local_contrast)

    display = np.clip(display, 0.0, 1.0)

    return ToneResult(
        image=display,
        window_mask=window_mask,
        shadow_mask=shadow_mask,
        applied_window_ev=window_ev,
        applied_shadow_ev=shadow_ev,
        exposure_gain=gain,
        stats={
            "window_area": float(window_mask.mean()),
            "shadow_area": float(shadow_mask.mean()),
        },
    )


def _filmic(linear: np.ndarray, *, white_point: float, desaturate: float = 0.85) -> np.ndarray:
    """Compresion de altas luces tipo Reinhard extendido, aplicada a la luminancia.

    Se mapea la luminancia y el color se arrastra proporcionalmente. Cerca del
    blanco, ademas, el color se lleva hacia neutro: un cielo muy brillante
    tiende a blanco en vez de saturar el canal azul hasta recortarlo. Sin este
    paso, la ventana sale como una mancha azul plana, que es la marca de fabrica
    del HDR mal resuelto.
    """
    lum = luminance(linear)
    safe = np.maximum(lum, EPS)
    w2 = max(white_point, 1e-3) ** 2
    mapped = safe * (1.0 + safe / w2) / (1.0 + safe)

    scaled = linear * (mapped / safe)[..., None]

    if desaturate > 0:
        # El disparador es el canal mas alto, no la luminancia: en un cielo azul
        # el canal azul llega a 1.0 mucho antes de que la luminancia se acerque,
        # y es ese canal el que se recorta y aplana el color.
        peak = scaled.max(axis=2)
        blend = (smoothstep(peak, 0.72, 1.05) * desaturate)[..., None]
        scaled = scaled * (1.0 - blend) + mapped[..., None] * blend

    return srgb_encode(np.clip(scaled, 0.0, 1.0))


def _local_contrast(display: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Microcontraste restringido a una mascara.

    Dentro de la ventana revela el paisaje; fuera no se toca nada, asi que no
    aparece el cerco luminoso tipico del HDR mal hecho.
    """
    sigma = feather_radius(display.shape, 0.012)
    blurred = gaussian(display, sigma)
    detail = display - blurred
    return np.clip(display + detail * (amount * mask)[..., None], 0.0, 1.0)


def estimate_window_mask(linear: np.ndarray, cfg) -> np.ndarray:
    """Mascara de ventana calculada antes del balance de blancos.

    El balance de blancos necesita saber que es ventana y que es interior para
    corregir por separado las dos luces, pero la mascara definitiva se calcula
    dentro de `map_tone`, que va despues. Esta version previa rompe esa
    dependencia circular: es la misma medida sobre la imagen todavia sin
    corregir, que a estos efectos da igual porque la luminancia apenas cambia.
    """
    window_cfg = cfg.tone.window_pull
    if not window_cfg.enabled:
        return np.zeros(linear.shape[:2], dtype=np.float32)

    working = linear * auto_exposure(
        linear,
        target_midtone=cfg.tone.target_midtone,
        midtone_percentile=cfg.tone.midtone_percentile,
    )
    log_lum = _log_luminance(working)
    anchor = float(np.log2(max(interior_level(luminance(working)), 1e-4)))
    low, high = window_cfg.stops_above_interior
    return soft_mask(
        log_lum,
        _normalized_guide(log_lum),
        anchor + float(low),
        anchor + float(high),
        window_cfg.feather,
    )
