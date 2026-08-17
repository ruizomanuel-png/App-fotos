"""Borrado automatico de personas, mascotas y coches.

Esto solo cubre lo que un modelo sabe reconocer por si solo. Un cable suelto,
un enchufe o el reflejo del tripode en un espejo no se detectan de forma fiable
-- no hay modelo que sepa que "sobran" en tu foto -- y quedan para el editor de
borrado manual de la v2.

El modulo funciona sin los modelos instalados: en ese caso no borra nada y lo
deja anotado, en lugar de romper el lote.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from .imageops import from_uint8, to_uint8

log = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    confidence: float
    area: float                       # fraccion del encuadre
    mask: np.ndarray | None = None    # bool (H, W)
    box: tuple[int, int, int, int] | None = None


@dataclass
class CleanupResult:
    image: np.ndarray
    removed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    backend: str = "none"


# --------------------------------------------------------------------------
# Deteccion
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_detector():
    """Carga YOLOv8n-seg una sola vez por proceso.

    Se usa la variante de segmentacion y no la de deteccion: una caja
    rectangular alrededor de una persona obliga a rellenar mucha mas superficie
    de la necesaria, y el relleno se nota. La mascara sigue la silueta.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return None, "ultralytics no instalado"

    try:
        return YOLO("yolov8n-seg.pt"), "yolov8n-seg"
    except Exception as exc:  # noqa: BLE001 - primera descarga sin red, permisos, etc.
        return None, f"no se pudo cargar el modelo de deteccion: {exc}"


def detect_objects(display: np.ndarray, classes: list[str], confidence: float) -> tuple[list[Detection], str]:
    """Busca en la imagen los objetos de las clases pedidas."""
    model, backend = _load_detector()
    if model is None:
        return [], backend

    wanted = {name.lower() for name in classes}
    height, width = display.shape[:2]

    results = model.predict(
        to_uint8(display)[:, :, ::-1],   # ultralytics espera BGR
        conf=float(confidence),
        verbose=False,
    )

    detections: list[Detection] = []
    for result in results:
        names = result.names
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None:
            continue

        for index in range(len(boxes)):
            label = str(names[int(boxes.cls[index])]).lower()
            if label not in wanted:
                continue

            mask = None
            if masks is not None and masks.data is not None and index < len(masks.data):
                raw = masks.data[index].cpu().numpy().astype(np.float32)
                mask = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR) > 0.5

            x1, y1, x2, y2 = (int(round(v)) for v in boxes.xyxy[index].tolist())
            if mask is None:
                mask = np.zeros((height, width), dtype=bool)
                mask[max(y1, 0) : y2, max(x1, 0) : x2] = True

            detections.append(
                Detection(
                    label=label,
                    confidence=float(boxes.conf[index]),
                    area=float(mask.mean()),
                    mask=mask,
                    box=(x1, y1, x2, y2),
                )
            )

    return detections, backend


# --------------------------------------------------------------------------
# Relleno
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_inpainter():
    try:
        from simple_lama_inpainting import SimpleLama

        return SimpleLama(), "lama"
    except Exception as exc:  # noqa: BLE001 - sin el paquete o sin poder descargar pesos
        log.info("LaMa no disponible (%s); se usara el relleno clasico de OpenCV", exc)
        return None, "opencv"


def inpaint(display: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, str]:
    """Rellena la zona enmascarada con el fondo mas plausible."""
    model, backend = _load_inpainter()
    mask_u8 = (mask.astype(np.uint8)) * 255

    if model is not None:
        from PIL import Image

        result = model(Image.fromarray(to_uint8(display)), Image.fromarray(mask_u8))
        filled = from_uint8(np.array(result.convert("RGB")))
        if filled.shape[:2] != display.shape[:2]:
            filled = cv2.resize(
                filled, (display.shape[1], display.shape[0]), interpolation=cv2.INTER_LANCZOS4
            )
        # Solo se sustituye dentro de la mascara: fuera, el pixel original.
        blend = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)[..., None]
        return np.clip(display * (1.0 - blend) + filled * blend, 0.0, 1.0), backend

    # Sin LaMa: relleno por propagacion de OpenCV. Sirve para objetos pequenos
    # sobre fondo uniforme y poco mas, pero es mejor que dejar el objeto.
    filled = cv2.inpaint(to_uint8(display), mask_u8, 5, cv2.INPAINT_TELEA)
    return from_uint8(filled), backend


# --------------------------------------------------------------------------
# Punto de entrada
# --------------------------------------------------------------------------


def _dilate(mask: np.ndarray, fraction: float) -> np.ndarray:
    """Ensancha la mascara. El borde de una silueta arrastra sombra y reflejo,
    y dejarlos produce un halo con la forma del objeto borrado."""
    if fraction <= 0:
        return mask
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    extent = max(ys.max() - ys.min(), xs.max() - xs.min())
    radius = max(3, int(round(extent * fraction)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def remove_objects(display: np.ndarray, cfg) -> CleanupResult:
    """Detecta y borra personas, mascotas y coches del encuadre."""
    cleanup_cfg = cfg.cleanup
    result = CleanupResult(image=display)

    if not cleanup_cfg.enabled:
        return result

    detections, backend = detect_objects(
        display, list(cleanup_cfg.classes), float(cleanup_cfg.confidence)
    )
    result.backend = backend

    if not detections:
        if backend not in {"yolov8n-seg"}:
            result.warnings.append(
                f"borrado automatico inactivo: {backend}. "
                "Instala el extra 'ai' para activarlo (pip install -e '.[ai]')"
            )
        return result

    max_area = float(cleanup_cfg.max_object_area)
    combined = np.zeros(display.shape[:2], dtype=bool)

    for detection in detections:
        if detection.area > max_area:
            # Borrar algo asi de grande obliga a inventarse media habitacion.
            # Mejor mandarla a revisar que entregar una alucinacion.
            reason = (
                f"hay un objeto '{detection.label}' que ocupa el {detection.area:.0%} "
                "del encuadre: demasiado grande para borrarlo sin inventar fondo"
            )
            result.blocking.append(reason)
            result.skipped.append(detection.label)
            continue
        combined |= _dilate(detection.mask, float(cleanup_cfg.mask_dilate))
        result.removed.append(detection.label)

    if combined.any():
        result.image, inpaint_backend = inpaint(display, combined)
        result.backend = f"{backend}+{inpaint_backend}"
        if inpaint_backend == "opencv":
            result.warnings.append(
                "relleno con OpenCV (LaMa no instalado): revisa las zonas borradas"
            )

    return result
