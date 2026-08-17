"""Lectura de archivos RAW (Sony ARW, Apple ProRAW DNG) a radiancia lineal.

Todo el pipeline trabaja en lineal, asi que el revelado se hace con gamma 1.0
y sin ajuste automatico de brillo: queremos los datos del sensor tal cual,
no la interpretacion que haga LibRaw de ellos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import rawpy

from .imageops import from_uint8, resize_long_side, srgb_decode

log = logging.getLogger(__name__)

RAW_EXTENSIONS = {
    ".arw", ".sr2", ".srf",           # Sony
    ".dng",                            # Apple ProRAW, Adobe DNG
    ".cr2", ".cr3", ".crw",            # Canon
    ".nef", ".nrw",                    # Nikon
    ".raf", ".orf", ".rw2", ".pef",    # Fuji, Olympus, Panasonic, Pentax
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | IMAGE_EXTENSIONS

_DEMOSAIC = {
    "AHD": rawpy.DemosaicAlgorithm.AHD,
    "VNG": rawpy.DemosaicAlgorithm.VNG,
    "PPG": rawpy.DemosaicAlgorithm.PPG,
    "DHT": rawpy.DemosaicAlgorithm.DHT,
    "LINEAR": rawpy.DemosaicAlgorithm.LINEAR,
}


@dataclass
class Frame:
    """Una toma del bracket, ya revelada a radiancia lineal."""

    path: Path
    image: np.ndarray                      # float32 (H, W, 3), lineal, RGB
    exposure_time: float | None = None     # segundos
    f_number: float | None = None
    iso: float | None = None
    timestamp: datetime | None = None
    camera: str | None = None
    lens: str | None = None
    focal_35mm: float | None = None
    white_balance: Sequence[float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def exposure_scale(self) -> float | None:
        """Factor que relaciona el valor del pixel con la radiancia de la escena.

        El valor digital es proporcional a ``L * t * ISO / N^2``, asi que
        dividir por este factor deja todas las tomas en la misma escala fisica.
        """
        if not self.exposure_time or not self.iso:
            return None
        aperture = self.f_number or 1.0
        scale = self.exposure_time * self.iso / max(aperture * aperture, 1e-6)
        return scale if scale > 0 else None

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]


# --------------------------------------------------------------------------
# EXIF
# --------------------------------------------------------------------------


def _ratio_to_float(value: Any) -> float | None:
    """Convierte los racionales de EXIF (`1/125`, `28/10`) a float."""
    if value is None:
        return None
    try:
        values = getattr(value, "values", None)
        if values:
            first = values[0]
            num = getattr(first, "num", None)
            den = getattr(first, "den", None)
            if num is not None and den:
                return float(num) / float(den)
            return float(first)
        return float(str(value))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _tag_to_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def read_exif(path: Path) -> dict[str, Any]:
    """Lee los campos EXIF que necesita el pipeline. Nunca lanza excepcion."""
    info: dict[str, Any] = {}
    try:
        import exifread
    except ImportError:  # pragma: no cover - exifread es dependencia declarada
        log.warning("exifread no disponible; se estimara la exposicion desde los pixeles")
        return info

    try:
        with path.open("rb") as handle:
            tags = exifread.process_file(handle, details=False)
    except Exception as exc:  # noqa: BLE001 - un EXIF roto no debe tumbar el lote
        log.warning("no se pudo leer el EXIF de %s: %s", path.name, exc)
        return info

    info["exposure_time"] = _ratio_to_float(tags.get("EXIF ExposureTime"))
    info["f_number"] = _ratio_to_float(tags.get("EXIF FNumber"))
    info["iso"] = _ratio_to_float(tags.get("EXIF ISOSpeedRatings")) or _ratio_to_float(
        tags.get("EXIF PhotographicSensitivity")
    )
    info["lens"] = _tag_to_str(tags.get("EXIF LensModel"))
    # La focal equivalente da la distancia focal en pixeles, y con ella la
    # correccion de perspectiva deja de ser una estimacion a ojo.
    info["focal_35mm"] = _ratio_to_float(tags.get("EXIF FocalLengthIn35mmFilm"))
    make = _tag_to_str(tags.get("Image Make")) or ""
    model = _tag_to_str(tags.get("Image Model")) or ""
    info["camera"] = f"{make} {model}".strip() or None

    raw_stamp = _tag_to_str(tags.get("EXIF DateTimeOriginal")) or _tag_to_str(
        tags.get("Image DateTime")
    )
    if raw_stamp:
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                info["timestamp"] = datetime.strptime(raw_stamp, fmt)
                break
            except ValueError:
                continue
    # Subsegundos: sin esto, tres disparos en rafaga comparten el mismo segundo
    # y no hay forma de ordenarlos.
    subsec = _tag_to_str(tags.get("EXIF SubSecTimeOriginal"))
    if subsec and info.get("timestamp"):
        try:
            fraction = float(f"0.{subsec}")
            info["timestamp"] = info["timestamp"].replace(microsecond=int(fraction * 1e6))
        except ValueError:
            pass

    return info


# --------------------------------------------------------------------------
# Revelado
# --------------------------------------------------------------------------


def _load_non_raw(path: Path) -> tuple[np.ndarray, None]:
    """Carga JPEG/PNG/TIFF y lo linealiza. Usado en pruebas y material ya revelado."""
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise ValueError(f"no se pudo abrir la imagen {path}")
    if data.ndim == 2:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    if data.shape[2] == 4:
        data = data[:, :, :3]
    if data.dtype == np.uint16:
        display = (data.astype(np.float32) / 65535.0)
    elif data.dtype == np.uint8:
        display = from_uint8(data)
    else:
        display = data.astype(np.float32)
    rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    return srgb_decode(rgb), None


def _load_raw(
    path: Path,
    *,
    use_camera_wb: bool,
    demosaic: str,
    user_wb: Sequence[float] | None,
) -> tuple[np.ndarray, list[float]]:
    with rawpy.imread(str(path)) as raw:
        camera_wb = [float(v) for v in raw.camera_whitebalance]
        params: dict[str, Any] = {
            "gamma": (1.0, 1.0),        # sin gamma: queremos lineal
            "no_auto_bright": True,     # sin reencuadre de niveles
            "output_bps": 16,
            "output_color": rawpy.ColorSpace.sRGB,
            "demosaic_algorithm": _DEMOSAIC.get(demosaic.upper(), rawpy.DemosaicAlgorithm.AHD),
        }
        # Todas las tomas del bracket comparten multiplicadores de balance de
        # blancos. Si cada una usara los suyos, la fusion mezclaria colores
        # ligeramente distintos y aparecerian dominantes por zonas.
        if user_wb is not None:
            params["user_wb"] = list(user_wb)
        elif use_camera_wb:
            params["use_camera_wb"] = True
        else:
            params["use_auto_wb"] = True

        rgb16 = raw.postprocess(**params)

    return (rgb16.astype(np.float32) / 65535.0), camera_wb


def load_frame(
    path: Path,
    *,
    use_camera_wb: bool = True,
    demosaic: str = "AHD",
    user_wb: Sequence[float] | None = None,
    max_long_side: int | None = None,
) -> Frame:
    """Revela un archivo a radiancia lineal y adjunta su EXIF."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in RAW_EXTENSIONS:
        image, camera_wb = _load_raw(
            path, use_camera_wb=use_camera_wb, demosaic=demosaic, user_wb=user_wb
        )
    elif suffix in IMAGE_EXTENSIONS:
        image, camera_wb = _load_non_raw(path)
    else:
        raise ValueError(f"extension no soportada: {path.name}")

    image = resize_long_side(np.ascontiguousarray(image, dtype=np.float32), max_long_side)

    exif = read_exif(path)
    return Frame(
        path=path,
        image=image,
        exposure_time=exif.get("exposure_time"),
        f_number=exif.get("f_number"),
        iso=exif.get("iso"),
        timestamp=exif.get("timestamp"),
        camera=exif.get("camera"),
        lens=exif.get("lens"),
        focal_35mm=exif.get("focal_35mm"),
        white_balance=camera_wb,
    )


def load_bracket(
    paths: Sequence[Path],
    *,
    use_camera_wb: bool = True,
    demosaic: str = "AHD",
    max_long_side: int | None = None,
) -> list[Frame]:
    """Revela un bracket completo con el mismo balance de blancos en todas las tomas."""
    if not paths:
        return []

    first = load_frame(
        paths[0],
        use_camera_wb=use_camera_wb,
        demosaic=demosaic,
        max_long_side=max_long_side,
    )
    shared_wb = first.white_balance if Path(paths[0]).suffix.lower() in RAW_EXTENSIONS else None

    frames = [first]
    for path in paths[1:]:
        frames.append(
            load_frame(
                path,
                use_camera_wb=use_camera_wb,
                demosaic=demosaic,
                user_wb=shared_wb,
                max_long_side=max_long_side,
            )
        )

    shapes = {frame.shape for frame in frames}
    if len(shapes) > 1:
        raise ValueError(
            "las tomas del bracket no tienen el mismo tamano: "
            + ", ".join(f"{f.path.name}={f.shape}" for f in frames)
        )
    return frames
