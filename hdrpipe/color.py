"""Balance de blancos y acabado de color.

Dos bloques distintos:

* `white_balance` trabaja en lineal, antes del mapeo de tono. Es donde se
  arregla el problema clasico del interiorismo: bombillas calidas dentro y luz
  de dia fria entrando por la ventana, en la misma foto.
* `grade` trabaja en espacio de pantalla, despues. Es la curva, la saturacion
  y los realces por color.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .imageops import (
    EPS,
    apply_lut,
    feather_radius,
    gaussian,
    luminance,
    robust_percentiles,
)


@dataclass
class ColorStats:
    illuminant: list[float] = field(default_factory=list)
    interior_illuminant: list[float] | None = None
    window_illuminant: list[float] | None = None


# --------------------------------------------------------------------------
# Balance de blancos
# --------------------------------------------------------------------------


def estimate_illuminant(
    linear: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    norm: float = 6.0,
) -> np.ndarray:
    """Color de la luz de la escena, normalizado a verde = 1.

    Implementa "shades of grey" (norma de Minkowski). Con norma 6 se queda entre
    el gris medio, que falla cuando hay una pared roja enorme, y el parche
    blanco, que falla en cuanto hay un reflejo especular.
    """
    pixels = linear.reshape(-1, 3)

    if mask is not None:
        selection = mask.reshape(-1) > 0.5
        if selection.sum() > 2000:
            pixels = pixels[selection]

    lum = pixels @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    # Fuera el ruido de las sombras profundas y los especulares quemados: ni
    # unos ni otros dicen nada sobre el color de la luz.
    usable = (lum > 0.01) & (pixels.max(axis=1) < 0.98)
    if usable.sum() > 2000:
        pixels = pixels[usable]

    if pixels.shape[0] > 300_000:
        pixels = pixels[:: pixels.shape[0] // 300_000 + 1]
    if pixels.shape[0] < 100:
        return np.ones(3, dtype=np.float32)

    powered = np.power(np.clip(pixels, 0.0, None), norm).mean(axis=0)
    illuminant = np.power(powered, 1.0 / norm).astype(np.float32)
    illuminant = np.maximum(illuminant, EPS)
    return (illuminant / illuminant[1]).astype(np.float32)


def _correction_gains(illuminant: np.ndarray, strength: float) -> np.ndarray:
    """Ganancias por canal que neutralizan un iluminante, aplicadas parcialmente."""
    full = 1.0 / np.maximum(illuminant, EPS)
    gains = np.power(full, float(np.clip(strength, 0.0, 1.0)))
    return (gains / gains[1]).astype(np.float32)


def white_balance(linear: np.ndarray, window_mask: np.ndarray | None, cfg) -> tuple[np.ndarray, ColorStats]:
    """Neutraliza la dominante global y, si procede, la mezcla interior/ventana."""
    wb_cfg = cfg.white_balance
    stats = ColorStats()

    if not wb_cfg.enabled:
        return linear, stats

    illuminant = estimate_illuminant(linear)
    stats.illuminant = [float(v) for v in illuminant]
    result = linear * _correction_gains(illuminant, wb_cfg.strength)

    split_cfg = wb_cfg.split_mixed_cast
    if split_cfg.enabled and window_mask is not None and window_mask.mean() > 0.005:
        interior_mask = 1.0 - window_mask
        if interior_mask.mean() > 0.05:
            window_illuminant = estimate_illuminant(result, window_mask)
            interior_illuminant = estimate_illuminant(result, interior_mask)
            stats.window_illuminant = [float(v) for v in window_illuminant]
            stats.interior_illuminant = [float(v) for v in interior_illuminant]

            window_gains = _correction_gains(window_illuminant, split_cfg.strength)
            interior_gains = _correction_gains(interior_illuminant, split_cfg.strength)

            # Los dos juegos de ganancias se mezclan con la propia mascara, que
            # ya viene suavizada por el guided filter. Sin ese degradado se veria
            # el salto de color justo en el marco de la ventana.
            blend = window_mask[..., None]
            result = result * (window_gains * blend + interior_gains * (1.0 - blend))

    # Una casa perfectamente neutra se ve fria. Se devuelve algo de calidez.
    warmth = float(wb_cfg.keep_warmth)
    if warmth > 0:
        result = result * np.array(
            [1.0 + 0.18 * warmth, 1.0, 1.0 - 0.16 * warmth], dtype=np.float32
        )

    return np.clip(result, 0.0, None).astype(np.float32), stats


# --------------------------------------------------------------------------
# Curva de tono
# --------------------------------------------------------------------------


def build_tone_curve(
    *,
    contrast: float,
    black_point: float,
    white_point: float,
    size: int = 1024,
) -> np.ndarray:
    """LUT monotona con curva en S suave.

    La mezcla con smoothstep garantiza que la curva nunca se invierte, por
    fuerte que sea el contraste; una curva no monotona produce posterizacion.
    """
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)

    span = max(white_point - black_point, 1e-3)
    normalized = np.clip((x - black_point) / span, 0.0, 1.0)

    s_shape = normalized * normalized * (3.0 - 2.0 * normalized)
    amount = float(np.clip(contrast, 0.0, 1.0))
    return ((1.0 - amount) * normalized + amount * s_shape).astype(np.float32)


# --------------------------------------------------------------------------
# Ajustes en espacio de pantalla
# --------------------------------------------------------------------------


def _apply_vibrance_saturation(rgb: np.ndarray, vibrance: float, saturation: float) -> np.ndarray:
    """Vibrance sube los colores apagados y respeta los que ya estan saturados."""
    if abs(vibrance) < 1e-4 and abs(saturation - 1.0) < 1e-4:
        return rgb

    gray = luminance(rgb)[..., None]
    chroma = rgb - gray

    if abs(vibrance) > 1e-4:
        current = np.abs(chroma).max(axis=2, keepdims=True) / np.maximum(gray, 0.05)
        factor = 1.0 + vibrance * (1.0 - np.clip(current, 0.0, 1.0))
        chroma = chroma * factor

    return np.clip(gray + chroma * saturation, 0.0, 1.0)


def _apply_targeted(rgb: np.ndarray, target) -> np.ndarray:
    """Realce restringido a un rango de tono (cesped, agua)."""
    if not target.enabled:
        return rgb

    hsv = cv2.cvtColor(np.clip(rgb, 0.0, 1.0), cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Distancia angular al centro del rango, teniendo en cuenta que 0 y 360
    # son el mismo color.
    delta = np.abs(((hue - float(target.hue_center) + 180.0) % 360.0) - 180.0)
    weight = np.clip(1.0 - delta / max(float(target.hue_width), 1e-3), 0.0, 1.0)
    weight = weight * weight * (3.0 - 2.0 * weight)
    # Un pixel casi gris no tiene tono fiable: retocarlo generaria manchas.
    weight = weight * np.clip(sat / 0.15, 0.0, 1.0)

    if abs(float(target.hue_shift)) > 1e-4:
        hsv[..., 0] = (hue + float(target.hue_shift) * weight) % 360.0
    hsv[..., 1] = np.clip(sat * (1.0 + (float(target.saturation) - 1.0) * weight), 0.0, 1.0)
    hsv[..., 2] = np.clip(val * (1.0 + (float(target.luminance) - 1.0) * weight), 0.0, 1.0)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _apply_clarity(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Contraste local en medios tonos. No toca ni negros ni altas luces."""
    if amount <= 1e-4:
        return rgb
    sigma = feather_radius(rgb.shape, 0.02)
    detail = rgb - gaussian(rgb, sigma)
    lum = luminance(rgb)
    midtones = (1.0 - np.abs(lum * 2.0 - 1.0))[..., None]
    return np.clip(rgb + detail * amount * midtones, 0.0, 1.0)


def grade(display: np.ndarray, cfg, *, curve: np.ndarray | None = None) -> np.ndarray:
    """Acabado final sobre la imagen ya mapeada a pantalla."""
    grade_cfg = cfg.grade

    if curve is None:
        # Si hay curva aprendida de tus pares antes/despues, manda esa: describe
        # tu look mejor que cualquier valor de contraste que se pueda poner a ojo.
        learned = grade_cfg.get("curve_lut")
        if learned:
            curve = np.asarray(learned, dtype=np.float32)
        else:
            curve = build_tone_curve(
                contrast=grade_cfg.contrast,
                black_point=grade_cfg.black_point,
                white_point=grade_cfg.white_point,
            )

    result = apply_lut(display, curve)
    result = _apply_clarity(result, float(grade_cfg.clarity))
    result = _apply_vibrance_saturation(
        result, float(grade_cfg.vibrance), float(grade_cfg.saturation)
    )
    result = _apply_targeted(result, grade_cfg.targeted.grass)
    result = _apply_targeted(result, grade_cfg.targeted.water)

    return np.clip(result, 0.0, 1.0)


# --------------------------------------------------------------------------
# Estadisticas para la consistencia del lote
# --------------------------------------------------------------------------


def image_signature(display: np.ndarray) -> dict[str, float]:
    """Resumen de tono y color de una foto, para comparar unas con otras."""
    lum = luminance(display)
    p05, p50, p95 = robust_percentiles(lum, [5.0, 50.0, 95.0])
    mean_rgb = display.reshape(-1, 3).mean(axis=0)
    green = max(float(mean_rgb[1]), EPS)
    return {
        "shadow": p05,
        "midtone": p50,
        "highlight": p95,
        "r_over_g": float(mean_rgb[0]) / green,
        "b_over_g": float(mean_rgb[2]) / green,
    }
