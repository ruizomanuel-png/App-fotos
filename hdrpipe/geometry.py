"""Geometria: distorsion de lente, verticales rectas y recorte.

En interiorismo la perspectiva es media edicion. Una foto con las paredes
inclinadas se ve amateur por muy bien resuelto que este el HDR. La correccion
se deduce del punto de fuga vertical de la propia escena: en un interior sobran
lineas verticales (marcos de puerta, esquinas, muebles, jambas de ventana).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .imageops import luminance, to_uint8

log = logging.getLogger(__name__)


@dataclass
class GeometryResult:
    image: np.ndarray
    applied_degrees: float = 0.0
    area_loss: float = 0.0
    lines_used: int = 0
    method: str = "none"                       # none | verticals | horizon
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Deteccion de lineas
# --------------------------------------------------------------------------


def detect_segments(display: np.ndarray, *, max_side: int = 1400) -> np.ndarray:
    """Segmentos rectos de la escena, en coordenadas de resolucion completa."""
    gray = to_uint8(np.clip(luminance(display), 0.0, 1.0))
    h, w = gray.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(
            gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )

    try:
        detector = cv2.createLineSegmentDetector()
        lines = detector.detect(gray)[0]
    except cv2.error as exc:  # pragma: no cover - depende del build de OpenCV
        log.warning("detector de segmentos no disponible: %s", exc)
        return np.empty((0, 4), dtype=np.float32)

    if lines is None or len(lines) == 0:
        return np.empty((0, 4), dtype=np.float32)

    return (lines.reshape(-1, 4) / scale).astype(np.float32)


def _segment_angles(segments: np.ndarray) -> np.ndarray:
    """Angulo de cada segmento respecto a la vertical de la imagen, en grados."""
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    return np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))


def _segment_lengths(segments: np.ndarray) -> np.ndarray:
    return np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])


# --------------------------------------------------------------------------
# Punto de fuga vertical
# --------------------------------------------------------------------------


def _homogeneous_lines(segments: np.ndarray) -> np.ndarray:
    ones = np.ones((len(segments), 1), dtype=np.float64)
    p1 = np.hstack([segments[:, 0:2].astype(np.float64), ones])
    p2 = np.hstack([segments[:, 2:4].astype(np.float64), ones])
    return np.cross(p1, p2)


def estimate_vertical_vp(
    segments: np.ndarray,
    shape: tuple[int, int],
    *,
    min_lines: int = 6,
    iterations: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray | None, int]:
    """Punto de fuga de las verticales, por RANSAC sobre pares de lineas.

    RANSAC y no minimos cuadrados directos porque en un salon hay lineas
    "verticales" que no lo son: el respaldo de una silla, el borde de una
    cortina. Con ajuste directo esas lineas arrastran el punto de fuga.
    """
    height, width = shape[:2]
    angles = _segment_angles(segments)
    lengths = _segment_lengths(segments)

    # Solo lineas razonablemente verticales y suficientemente largas.
    keep = (angles < 30.0) & (lengths > 0.06 * height)
    candidates = segments[keep]
    if len(candidates) < min_lines:
        return None, len(candidates)

    lines = _homogeneous_lines(candidates)
    # Normalizar cada linea deja la distancia punto-recta en unidades de pixel.
    norms = np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    lines = lines / np.maximum(norms, 1e-9)

    weights = _segment_lengths(candidates)
    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    best_score = -1.0
    tolerance = 0.004 * max(height, width)

    for _ in range(iterations):
        i, j = rng.choice(len(lines), size=2, replace=False)
        vp = np.cross(lines[i], lines[j])
        if abs(vp[2]) < 1e-12:
            continue  # punto de fuga en el infinito: verticales ya paralelas
        vp = vp / vp[2]

        # Un punto de fuga vertical plausible cae muy por encima o por debajo
        # del encuadre. Si cae dentro, el ajuste es espurio.
        if abs(vp[1] - height / 2) < height * 0.9:
            continue

        distance = np.abs(lines @ vp) / max(np.linalg.norm(vp[:2]), 1e-9)
        inliers = distance < tolerance
        score = float(weights[inliers].sum())
        if score > best_score:
            best_score, best_inliers = score, inliers

    if best_inliers is None or int(best_inliers.sum()) < min_lines:
        return None, int(best_inliers.sum()) if best_inliers is not None else 0

    # Refinado sobre los inliers: el punto que minimiza la distancia a todas
    # las lineas es el vector singular menor de la matriz de lineas.
    selected = lines[best_inliers] * weights[best_inliers][:, None]
    _, _, vt = np.linalg.svd(selected)
    vp = vt[-1]
    if abs(vp[2]) < 1e-12:
        return None, int(best_inliers.sum())

    return (vp / vp[2]), int(best_inliers.sum())


def _intrinsics(shape: tuple[int, int], focal_35mm: float | None) -> np.ndarray:
    height, width = shape[:2]
    if focal_35mm and focal_35mm > 0:
        # 36 mm es el lado largo del fotograma de 35 mm.
        focal_px = (max(width, height) / 36.0) * focal_35mm
    else:
        focal_px = 1.2 * max(width, height)
    return np.array(
        [[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rectifying_homography(
    vp: np.ndarray,
    shape: tuple[int, int],
    *,
    focal_35mm: float | None = None,
    strength: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Homografia que endereza las verticales. Devuelve (H, grados aplicados).

    Se pasa el punto de fuga a direccion en coordenadas de camara, se calcula la
    rotacion que la lleva a la vertical de la imagen y se aplica una fraccion de
    esa rotacion. Al ser una rotacion pura de camara, la correccion es la misma
    que se obtendria habiendo nivelado el tripode: no deforma la escena.
    """
    K = _intrinsics(shape, focal_35mm)
    direction = np.linalg.inv(K) @ vp
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.eye(3), 0.0
    direction = direction / norm

    target = np.array([0.0, -1.0, 0.0])          # "arriba" en coordenadas de imagen
    if direction @ target < 0:
        direction = -direction                    # el punto de fuga puede salir invertido

    axis = np.cross(direction, target)
    sin_angle = np.linalg.norm(axis)
    cos_angle = float(np.clip(direction @ target, -1.0, 1.0))
    angle = math.atan2(sin_angle, cos_angle)

    if sin_angle < 1e-9 or angle < 1e-6:
        return np.eye(3), 0.0

    axis = axis / sin_angle
    applied = angle * float(np.clip(strength, 0.0, 1.0))

    # Rodrigues: rotacion de `applied` radianes alrededor de `axis`.
    R, _ = cv2.Rodrigues(axis * applied)
    H = K @ R @ np.linalg.inv(K)
    return H / H[2, 2], math.degrees(applied)


def estimate_roll(segments: np.ndarray, shape: tuple[int, int], max_degrees: float) -> float:
    """Inclinacion de la camara a partir de las lineas horizontales dominantes."""
    height, width = shape[:2]
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    lengths = np.hypot(dx, dy)

    angles = np.degrees(np.arctan2(dy, dx))
    angles = (angles + 90.0) % 180.0 - 90.0       # a [-90, 90)
    keep = (np.abs(angles) < max_degrees) & (lengths > 0.08 * width)
    if keep.sum() < 4:
        return 0.0

    # Mediana ponderada por longitud: una linea larga es mas fiable que el
    # borde de un cojin.
    order = np.argsort(angles[keep])
    sorted_angles = angles[keep][order]
    cumulative = np.cumsum(lengths[keep][order])
    midpoint = cumulative[-1] / 2.0
    return float(sorted_angles[int(np.searchsorted(cumulative, midpoint))])


# --------------------------------------------------------------------------
# Recorte
# --------------------------------------------------------------------------


def _clip_halfplane(subject: list[np.ndarray], a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    """Recorta un poligono por el semiplano a la izquierda del segmento a->b."""
    if not subject:
        return []
    edge = b - a

    def inside(point) -> bool:
        return edge[0] * (point[1] - a[1]) - edge[1] * (point[0] - a[0]) >= -1e-9

    output: list[np.ndarray] = []
    for index, point in enumerate(subject):
        previous = subject[index - 1]
        if inside(point):
            if not inside(previous):
                output.append(_line_intersection(previous, point, a, b))
            output.append(point)
        elif inside(previous):
            output.append(_line_intersection(previous, point, a, b))
    return output


def _clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Interseccion de dos poligonos convexos (Sutherland-Hodgman)."""
    output = list(subject)
    for index in range(len(clip)):
        output = _clip_halfplane(output, clip[index], clip[(index + 1) % len(clip)])
        if not output:
            return np.empty((0, 2))
    return np.array(output, dtype=np.float64)


def _feasible_centers(
    polygon: np.ndarray, half_width: float, half_height: float
) -> np.ndarray:
    """Centros validos para un rectangulo de ese tamano dentro del poligono.

    Cada lado del poligono se desplaza hacia dentro la distancia justa para que
    la esquina mas desfavorable del rectangulo siga cayendo del lado bueno. La
    interseccion de todos esos semiplanos es, de nuevo, un poligono convexo: si
    no esta vacio, el rectangulo cabe en alguna posicion.
    """
    minimum = polygon.min(axis=0) - 1.0
    maximum = polygon.max(axis=0) + 1.0
    region = [
        np.array([minimum[0], minimum[1]]),
        np.array([maximum[0], minimum[1]]),
        np.array([maximum[0], maximum[1]]),
        np.array([minimum[0], maximum[1]]),
    ]

    for index in range(len(polygon)):
        a = polygon[index]
        b = polygon[(index + 1) % len(polygon)]
        edge = b - a
        length = float(np.hypot(edge[0], edge[1]))
        if length < 1e-9:
            continue
        normal = np.array([-edge[1], edge[0]]) / length      # normal interior
        support = abs(normal[0]) * half_width + abs(normal[1]) * half_height
        shift = normal * support
        region = _clip_halfplane(region, a + shift, b + shift)
        if not region:
            return np.empty((0, 2))

    return np.array(region, dtype=np.float64)


def _line_intersection(p1, p2, p3, p4) -> np.ndarray:
    d1, d2 = p2 - p1, p4 - p3
    denominator = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denominator) < 1e-12:
        return p2
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denominator
    return p1 + t * d1


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def reframe_homography(
    homography: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, float]:
    """Anade al enderezado el reencuadre que aprovecha toda la zona util.

    Enderezar es una rotacion de camara, y una rotacion mueve el encuadre
    entero: parte de la imagen corregida cae fuera del lienzo original y, si se
    recorta contra ese lienzo, se tira contenido perfectamente valido. Aqui se
    busca el mayor rectangulo con el encuadre original dentro del cuadrilatero
    corregido -- sin referencia al lienzo de partida -- y se compone la
    transformacion que lo lleva al tamano de salida.

    Devuelve (homografia final, fraccion del original que se pierde).
    """
    height, width = shape[:2]
    corners = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64
    ).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    if _polygon_area(quad) < 1.0:
        return homography, 0.0

    best_scale = 0.0
    best_center = quad.mean(axis=0)

    # El rectangulo puede acabar siendo mayor que el original (la correccion
    # amplia parte de la escena), de ahi que la busqueda llegue hasta 2.0.
    low, high = 0.0, 2.0
    for _ in range(34):
        middle = (low + high) / 2.0
        centers = _feasible_centers(quad, width * middle / 2.0, height * middle / 2.0)
        if len(centers) >= 3:
            low, best_scale, best_center = middle, middle, centers.mean(axis=0)
        else:
            high = middle

    if best_scale <= 1e-3:
        return homography, 1.0

    crop_w, crop_h = width * best_scale, height * best_scale
    x, y = best_center[0] - crop_w / 2.0, best_center[1] - crop_h / 2.0

    # Lleva el rectangulo elegido al lienzo de salida completo.
    scale = width / crop_w
    reframe = np.array(
        [[scale, 0.0, -x * scale], [0.0, scale, -y * scale], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    final = reframe @ homography

    # Lo perdido se mide sobre la foto original: que parte del encuadre de
    # partida no sobrevive al recorte.
    rect = np.array(
        [[x, y], [x + crop_w, y], [x + crop_w, y + crop_h], [x, y + crop_h]], dtype=np.float64
    ).reshape(-1, 1, 2)
    preimage = cv2.perspectiveTransform(rect, np.linalg.inv(homography)).reshape(-1, 2)
    kept = _polygon_area(_clip_convex(preimage, corners.reshape(-1, 2)))
    area_loss = float(np.clip(1.0 - kept / (width * height), 0.0, 1.0))

    return final / final[2, 2], area_loss


# --------------------------------------------------------------------------
# Distorsion de lente
# --------------------------------------------------------------------------


def undistort(image: np.ndarray, profile, focal_35mm: float | None) -> np.ndarray:
    """Corrige la distorsion radial con un perfil calibrado a mano por objetivo."""
    K = _intrinsics(image.shape[:2], focal_35mm)
    coefficients = np.array(
        [
            float(profile.get("k1", 0.0)),
            float(profile.get("k2", 0.0)),
            0.0,
            0.0,
            float(profile.get("k3", 0.0)),
        ],
        dtype=np.float64,
    )
    if not np.any(coefficients):
        return image
    return cv2.undistort(image, K, coefficients)


def _match_lens_profile(profiles, lens: str | None) -> dict | None:
    if not lens or not profiles:
        return None
    lens_lower = lens.lower()
    for key, profile in profiles.items():
        if str(key).lower() in lens_lower:
            return dict(profile)
    return None


# --------------------------------------------------------------------------
# Punto de entrada
# --------------------------------------------------------------------------


def correct_geometry(
    display: np.ndarray,
    cfg,
    *,
    lens: str | None = None,
    focal_35mm: float | None = None,
) -> GeometryResult:
    """Aplica distorsion de lente, enderezado y recorte en un unico remuestreo."""
    geometry_cfg = cfg.geometry
    result = GeometryResult(image=display)

    profile = _match_lens_profile(geometry_cfg.lens_profiles, lens)
    working = undistort(display, profile, focal_35mm) if profile else display

    segments = detect_segments(working)
    if len(segments) == 0:
        result.image = working
        result.warnings.append("no se detectaron lineas rectas; se deja la geometria original")
        return result

    homography: np.ndarray | None = None
    vertical_cfg = geometry_cfg.verticals

    if vertical_cfg.enabled:
        vp, line_count = estimate_vertical_vp(
            segments, working.shape[:2], min_lines=int(vertical_cfg.min_lines)
        )
        result.lines_used = line_count
        if vp is not None:
            candidate, degrees = rectifying_homography(
                vp,
                working.shape[:2],
                focal_35mm=focal_35mm,
                strength=float(vertical_cfg.strength),
            )
            # El limite se compara contra la inclinacion detectada, no contra la
            # fraccion que se va a aplicar: lo que decide si la escena es rara
            # es como se disparo, no cuanto hayamos elegido corregir.
            detected = degrees / max(float(vertical_cfg.strength), 1e-3)
            if detected > float(vertical_cfg.max_degrees):
                result.warnings.append(
                    f"la camara estaba inclinada {detected:.1f} grados "
                    f"(limite {vertical_cfg.max_degrees}); se deja sin corregir"
                )
            elif degrees > 0.15:
                homography = candidate
                result.applied_degrees = degrees
                result.method = "verticals"

    if homography is None and geometry_cfg.straighten.enabled:
        roll = estimate_roll(
            segments, working.shape[:2], float(geometry_cfg.straighten.max_degrees)
        )
        if abs(roll) > 0.15:
            height, width = working.shape[:2]
            rotation = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), roll, 1.0)
            homography = np.vstack([rotation, [0.0, 0.0, 1.0]])
            result.applied_degrees = abs(roll)
            result.method = "horizon"

    if homography is None:
        result.image = working
        return result

    height, width = working.shape[:2]

    if geometry_cfg.autocrop.enabled:
        homography, area_loss = reframe_homography(homography, (height, width))
        result.area_loss = area_loss
        if area_loss > float(geometry_cfg.autocrop.max_area_loss):
            result.warnings.append(
                f"el enderezado obligaria a recortar el {area_loss:.0%} de la imagen; "
                "se deja sin corregir"
            )
            result.image = working
            result.applied_degrees = 0.0
            result.method = "none"
            return result

    # Enderezado y reencuadre van en la misma matriz, asi que la imagen se
    # remuestrea una sola vez. Corregir y despues recortar y reescalar suma tres
    # interpolaciones y se come el detalle fino.
    warped = cv2.warpPerspective(
        working,
        homography,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )

    result.image = np.clip(warped, 0.0, 1.0)
    return result
