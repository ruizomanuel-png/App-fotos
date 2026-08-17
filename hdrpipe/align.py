"""Alineacion de las tomas del bracket.

Con tripode las tres tomas casi coinciden, pero "casi" no basta: un
desplazamiento de dos pixeles al fusionar produce bordes dobles justo en los
marcos de ventana, que es donde mas se nota. Se estima solo traslacion, que es
lo unico que puede pasar en un tripode (el zoom y la rotacion implicarian que
alguien ha tocado la camara, y eso es motivo de revision, no de correccion).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .imageops import luminance
from .raw import Frame

log = logging.getLogger(__name__)


def _alignment_proxy(image: np.ndarray, max_side: int) -> np.ndarray:
    """Version reducida y comprimida en gamma, para comparar tomas de distinto brillo."""
    gray = luminance(np.clip(image, 0.0, 1.0))
    gray = np.power(gray, 1.0 / 2.2, dtype=np.float32)   # acerca las exposiciones
    h, w = gray.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(
            gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
    # Normalizar quita la diferencia de exposicion residual que ECC no absorbe.
    lo, hi = float(gray.min()), float(gray.max())
    if hi - lo > 1e-6:
        gray = (gray - lo) / (hi - lo)
    return np.ascontiguousarray(gray, dtype=np.float32), scale


def estimate_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    estimate_max_side: int = 1200,
) -> tuple[float, float] | None:
    """Desplazamiento (dx, dy) en pixeles a resolucion completa, o None si falla."""
    ref_small, scale = _alignment_proxy(reference, estimate_max_side)
    mov_small, _ = _alignment_proxy(moving, estimate_max_side)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    try:
        cv2.findTransformECC(
            ref_small, mov_small, warp, cv2.MOTION_TRANSLATION, criteria, None, 5
        )
    except cv2.error as exc:
        log.warning("ECC no convergio: %s", exc)
        return None

    if not np.all(np.isfinite(warp)):
        return None
    return float(warp[0, 2] / scale), float(warp[1, 2] / scale)


def _translate(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )


def align_frames(
    frames: list[Frame],
    *,
    enabled: bool = True,
    max_shift_px: float = 40.0,
    estimate_max_side: int = 1200,
    reference_index: int | None = None,
) -> tuple[list[Frame], list[str]]:
    """Alinea las tomas contra la de referencia. Devuelve (tomas, avisos)."""
    warnings: list[str] = []
    if not enabled or len(frames) < 2:
        return frames, warnings

    ref_idx = reference_index if reference_index is not None else len(frames) // 2
    reference = frames[ref_idx].image

    for index, frame in enumerate(frames):
        if index == ref_idx:
            continue
        shift = estimate_shift(reference, frame.image, estimate_max_side=estimate_max_side)
        if shift is None:
            warnings.append(f"no se pudo alinear {frame.path.name}")
            continue

        dx, dy = shift
        magnitude = float(np.hypot(dx, dy))
        if magnitude > max_shift_px:
            # Mover tanto significa que la camara se movio de verdad. Deformar la
            # toma solo empeoraria el resultado.
            warnings.append(
                f"{frame.path.name} esta desplazada {magnitude:.0f} px "
                f"(limite {max_shift_px:.0f}); se deja sin alinear"
            )
            continue
        if magnitude < 0.25:
            continue

        frame.image = _translate(frame.image, dx, dy)
        log.debug("%s alineada (%.2f, %.2f) px", frame.path.name, dx, dy)

    return frames, warnings
