"""Ruido y enfoque.

Fusionar tres tomas ya promedia buena parte del ruido, asi que aqui se aplica
poco y con cuidado. El riesgo real en interiorismo no es quedarse corto de
enfoque, es enfocar una pared lisa y convertir su ruido en textura.
"""

from __future__ import annotations

import cv2
import numpy as np

from .imageops import gaussian, smoothstep, to_uint8


def _denoise(display: np.ndarray, chroma: float, luma: float) -> np.ndarray:
    """Reduccion de ruido separando luminancia y color.

    El ruido de color es el que se ve (manchas moradas en las sombras) y se
    puede atacar fuerte porque el ojo no percibe detalle fino en color. El de
    luminancia lleva el detalle real y apenas se toca.
    """
    if chroma <= 0 and luma <= 0:
        return display

    lab = cv2.cvtColor(np.clip(display, 0.0, 1.0), cv2.COLOR_RGB2Lab)

    if chroma > 0:
        radius = max(1, int(round(3 * chroma)))
        for channel in (1, 2):
            lab[..., channel] = cv2.bilateralFilter(
                lab[..., channel], d=0, sigmaColor=8.0 * chroma, sigmaSpace=radius * 2.0
            )

    if luma > 0:
        # El filtro bilateral sobre L preserva bordes; la mezcla parcial evita
        # el aspecto de plastico.
        smoothed = cv2.bilateralFilter(lab[..., 0], d=0, sigmaColor=3.0 * luma, sigmaSpace=4.0)
        lab[..., 0] = lab[..., 0] * (1.0 - luma) + smoothed * luma

    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0.0, 1.0)


def _sharpen(display: np.ndarray, amount: float, radius: float, edge_threshold: float) -> np.ndarray:
    """Mascara de enfoque limitada a los bordes reales."""
    if amount <= 0:
        return display

    blurred = gaussian(display, radius)
    detail = display - blurred

    # Solo se enfoca donde hay estructura. En una pared lisa el gradiente es
    # ruido, y enfocarlo lo hace visible.
    gray = cv2.cvtColor(np.clip(display, 0.0, 1.0), cv2.COLOR_RGB2GRAY)
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    edges = smoothstep(gradient, edge_threshold, edge_threshold * 4.0)[..., None]

    return np.clip(display + detail * amount * edges, 0.0, 1.0)


def enhance(display: np.ndarray, cfg) -> np.ndarray:
    detail_cfg = cfg.detail
    result = _denoise(
        display, float(detail_cfg.chroma_denoise), float(detail_cfg.luma_denoise)
    )
    return _sharpen(
        result,
        float(detail_cfg.sharpen.amount),
        float(detail_cfg.sharpen.radius),
        float(detail_cfg.sharpen.edge_threshold),
    )


def sharpness_score(display: np.ndarray) -> float:
    """Varianza del laplaciano: la medida clasica de "que tan enfocada esta"."""
    gray = to_uint8(cv2.cvtColor(np.clip(display, 0.0, 1.0), cv2.COLOR_RGB2GRAY))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
