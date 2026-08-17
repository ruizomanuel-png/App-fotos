"""Operaciones de imagen de bajo nivel compartidas por todo el pipeline.

Convenios que se respetan en todo el paquete:

* Las imagenes son ``np.ndarray`` de ``float32``, forma ``(alto, ancho, 3)``, RGB.
* "lineal" = radiancia proporcional a la luz real. Sin limite superior:
  una ventana puede valer 40.0 mientras el sofa vale 0.05. Aqui es donde
  tienen sentido las mezclas de exposicion y el window pull.
* "pantalla" = ya con gamma sRGB aplicada, acotado a [0, 1]. Aqui es donde
  tienen sentido el contraste, la saturacion y el enfoque.
"""

from __future__ import annotations

import cv2
import numpy as np

# Coeficientes Rec.709: la luminancia percibida no reparte igual R, G y B.
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

EPS = 1e-6


# --------------------------------------------------------------------------
# Espacios de color
# --------------------------------------------------------------------------


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Lineal -> pantalla (curva sRGB). La entrada puede exceder 1.0."""
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(
        np.float32
    )


def srgb_decode(display: np.ndarray) -> np.ndarray:
    """Pantalla -> lineal (inversa de la curva sRGB)."""
    x = np.clip(display, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def luminance(image: np.ndarray) -> np.ndarray:
    """Luminancia de una imagen RGB. Devuelve ``(alto, ancho)``."""
    return np.tensordot(image, LUMA_WEIGHTS, axes=([-1], [0])).astype(np.float32)


def scale_luminance(image: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Multiplica la imagen por una ganancia por pixel conservando el tono."""
    return image * gain[..., None]


# --------------------------------------------------------------------------
# Filtros
# --------------------------------------------------------------------------


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float,
    subsample: int = 4,
) -> np.ndarray:
    """Guided filter de He et al., version rapida (subsampling).

    Es la pieza que hace que las mascaras de ventana y de sombra sigan los
    bordes reales de la escena. Sin esto, cualquier ajuste local deja el halo
    gris tipico alrededor de los marcos de ventana.

    `guide` y `src` son mapas 2D. `radius` va en pixeles a resolucion completa.
    """
    guide = guide.astype(np.float32, copy=False)
    src = src.astype(np.float32, copy=False)

    s = max(1, int(subsample))
    r = max(1, int(round(radius / s)))

    if s > 1:
        size = (max(1, guide.shape[1] // s), max(1, guide.shape[0] // s))
        g_small = cv2.resize(guide, size, interpolation=cv2.INTER_AREA)
        p_small = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
    else:
        g_small, p_small = guide, src

    ksize = (2 * r + 1, 2 * r + 1)
    mean_g = cv2.boxFilter(g_small, -1, ksize)
    mean_p = cv2.boxFilter(p_small, -1, ksize)
    corr_gg = cv2.boxFilter(g_small * g_small, -1, ksize)
    corr_gp = cv2.boxFilter(g_small * p_small, -1, ksize)

    var_g = corr_gg - mean_g * mean_g
    cov_gp = corr_gp - mean_g * mean_p

    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g

    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)

    if s > 1:
        full = (guide.shape[1], guide.shape[0])
        mean_a = cv2.resize(mean_a, full, interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, full, interpolation=cv2.INTER_LINEAR)

    return (mean_a * guide + mean_b).astype(np.float32)


def feather_radius(shape: tuple[int, ...], fraction: float) -> int:
    """Convierte un radio relativo al lado corto en pixeles."""
    short_side = min(shape[0], shape[1])
    return max(2, int(round(short_side * float(fraction))))


def gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))


# --------------------------------------------------------------------------
# Mascaras
# --------------------------------------------------------------------------


def smoothstep(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Rampa suave de 0 a 1 entre `low` y `high`, con derivada nula en los extremos."""
    if high <= low:
        return (x >= high).astype(np.float32)
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def robust_percentiles(values: np.ndarray, percentiles: list[float]) -> list[float]:
    """Percentiles calculados sobre una muestra: mucho mas rapido en 40 MP."""
    flat = values.reshape(-1)
    if flat.size > 400_000:
        step = flat.size // 400_000 + 1
        flat = flat[::step]
    return [float(v) for v in np.percentile(flat, percentiles)]


def soft_mask(
    values: np.ndarray,
    guide: np.ndarray,
    low: float,
    high: float,
    feather: float,
    *,
    invert: bool = False,
) -> np.ndarray:
    """Mascara suave entre dos umbrales, pegada a los bordes reales de la escena.

    Los umbrales son absolutos, no percentiles. Un percentil describe cuanta
    superficie se selecciona, y eso no sirve para encontrar ventanas: una
    ventana puede ocupar el 2% de una foto y el 30% de la siguiente. Lo que
    define una ventana es que esta varias paradas por encima del interior,
    ocupe lo que ocupe.
    """
    mask = smoothstep(values, low, high)
    if invert:
        mask = 1.0 - mask
    radius = feather_radius(values.shape, feather)
    return np.clip(guided_filter(guide, mask, radius, eps=1e-4), 0.0, 1.0)


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------


def resize_long_side(image: np.ndarray, max_long_side: int | None) -> np.ndarray:
    """Reduce la imagen para que su lado largo no supere `max_long_side`."""
    if not max_long_side:
        return image
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_long_side:
        return image
    scale = max_long_side / longest
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def apply_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Aplica una LUT 1D (valores en [0,1]) por interpolacion lineal."""
    x = np.clip(image, 0.0, 1.0)
    positions = np.linspace(0.0, 1.0, len(lut), dtype=np.float32)
    return np.interp(x, positions, lut).astype(np.float32)


def to_uint8(display: np.ndarray) -> np.ndarray:
    return np.clip(display * 255.0 + 0.5, 0, 255).astype(np.uint8)


def from_uint8(image: np.ndarray) -> np.ndarray:
    return (image.astype(np.float32) / 255.0).astype(np.float32)
