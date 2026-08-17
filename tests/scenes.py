"""Generador de escenas sinteticas para las pruebas.

Reproduce lo esencial de un interior fotografiado a contraluz: paredes en
penumbra, un rincon oscuro y una ventana varias paradas mas brillante con
vegetacion detras. Es lo que hace falta para comprobar que el window pull y el
levantado de sombras hacen su trabajo.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hdrpipe.imageops import srgb_encode

# Tiempos de un bracket de tres tomas a -2 / 0 / +2 EV.
SHUTTER_SPEEDS = (1 / 200, 1 / 50, 1 / 12.5)
BASE_SHUTTER = 1 / 50
SENSOR_GAIN = 3.6


def interior_scene(
    height: int = 400,
    width: int = 600,
    *,
    brightness: float = 1.0,
    warmth: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Radiancia de la escena. `brightness` y `warmth` desvian una escena
    respecto de otra, que es lo que la consistencia de lote debe corregir."""
    rng = np.random.default_rng(seed)
    scene = np.zeros((height, width, 3), np.float32)

    wall = np.array([0.030 * warmth, 0.020, 0.012 / warmth], dtype=np.float32)
    scene[:] = wall
    scene[int(height * 0.62) :] = wall * 0.55                      # suelo
    scene[int(height * 0.75) :, : int(width * 0.2)] = wall * 0.12  # rincon en sombra

    # Ventana con vegetacion, unas cinco paradas por encima del interior.
    top, bottom = int(height * 0.15), int(height * 0.65)
    left, right = int(width * 0.6), int(width * 0.93)
    scene[top:bottom, left:right] = [0.55, 0.65, 0.95]
    scene[top + 40 : top + 90, left + 40 : right - 40] = [0.22, 0.34, 0.15]

    # Lineas verticales: marcos y esquinas. Sin ellas no hay punto de fuga que
    # detectar y la correccion de perspectiva no tiene con que trabajar.
    for x in (int(width * 0.12), int(width * 0.35), left, right):
        cv2.line(scene, (x, 0), (x, height), (0.09, 0.075, 0.055), 3)

    scene *= brightness
    scene += rng.normal(0, 0.0012, scene.shape).astype(np.float32)
    return np.clip(scene, 0.0, None)


def write_bracket(directory: Path, name: str, scene: np.ndarray, seed: int = 0) -> list[Path]:
    """Escribe las tres tomas de una escena como TIFF de 16 bits.

    Se usa TIFF y no RAW porque no se puede sintetizar un ARW valido; el
    pipeline los lee por la misma ruta y deduce las exposiciones de los pixeles.
    """
    rng = np.random.default_rng(seed + 991)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []

    for index, shutter in enumerate(SHUTTER_SPEEDS):
        exposed = scene * (shutter / BASE_SHUTTER) * SENSOR_GAIN
        exposed = exposed + rng.normal(0, 0.0015, exposed.shape).astype(np.float32)
        encoded = srgb_encode(np.clip(exposed, 0.0, 1.0))
        data = (np.clip(encoded, 0, 1) * 65535).astype(np.uint16)
        path = directory / f"{name}_{index + 1}.tif"
        cv2.imwrite(str(path), cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
        paths.append(path)

    return paths


def write_batch(directory: Path, count: int = 3) -> Path:
    """Un lote de varias escenas con brillo y calidez ligeramente distintos."""
    for index in range(count):
        scene = interior_scene(
            brightness=1.0 + 0.28 * (index - count / 2) / max(count, 1),
            warmth=1.0 + 0.12 * index,
            seed=index,
        )
        write_bracket(directory, f"escena{index + 1:02d}", scene, seed=index)
    return directory
