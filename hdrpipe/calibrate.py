"""Calibracion del look a partir de tus pares antes/despues.

No entrena ningun modelo: con treinta pares no daria para eso. Lo que hace es
medir en tus exports de Lightroom como tratas el tono y el color, y ajustar los
parametros del pipeline para que salga lo mismo.

Se aprenden cuatro cosas:

* **Cuanto de clara** dejas una foto  -> `tone.target_midtone`
* **Tu curva de tono**, punto por punto -> `grade.curve_lut`
* **Cuanta calidez** conservas          -> `white_balance.keep_warmth`
* **Cuanto color** metes                -> `grade.saturation`

Estructura de carpetas esperada:

    calibracion/
        raw/     los brackets originales (se agrupan solos)
        final/   tus JPEG exportados, un archivo por escena
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .color import build_tone_curve, image_signature
from .config import Config, deep_merge
from .grouping import Bracket, group_brackets, list_supported_files
from .imageops import EPS, apply_lut, luminance, resize_long_side
from .raw import IMAGE_EXTENSIONS
from .tone import interior_level

log = logging.getLogger(__name__)

# Resolucion a la que se comparan las imagenes. La curva de tono es una
# estadistica global; no hace falta resolucion completa y asi cabe todo en RAM.
COMPARE_LONG_SIDE = 700
LUT_SIZE = 256
# Vueltas del bucle de refinado. A partir de la tercera el cambio es residual.
REFINE_ITERATIONS = 6
# Fraccion de la correccion que se aplica en cada vuelta. A plena ganancia,
# calidez y saturacion se persiguen la una a la otra y oscilan.
DAMPING = 0.6
# Gamma sRGB. Relaciona una ganancia aplicada en lineal con su efecto en pantalla.
GAMMA = 2.4
# Tope de calidez. En el extremo son ganancias lineales [1.18, 1, 0.84], que en
# pantalla se ven como [1.07, 1, 0.93]: calido de verdad, pero todavia creible.
WARMTH_LIMIT = 1.0


@dataclass
class CalibrationResult:
    pairs_used: int = 0
    target_midtone: float = 0.46
    curve: np.ndarray | None = None
    keep_warmth: float = 0.12
    saturation: float = 1.0
    error: float = float("inf")
    notes: list[str] = field(default_factory=list)

    def to_overrides(self) -> dict:
        overrides: dict = {
            "tone": {"target_midtone": round(float(self.target_midtone), 4)},
            "white_balance": {"keep_warmth": round(float(self.keep_warmth), 4)},
            "grade": {"saturation": round(float(self.saturation), 4)},
        }
        if self.curve is not None:
            overrides["grade"]["curve_lut"] = [round(float(v), 5) for v in self.curve]
        return overrides


# --------------------------------------------------------------------------
# Emparejado
# --------------------------------------------------------------------------


def find_pairs(root: Path) -> list[tuple[Bracket, Path]]:
    """Empareja cada bracket de `raw/` con su export en `final/`."""
    root = Path(root)
    raw_dir = root / "raw" if (root / "raw").is_dir() else root
    final_dir = root / "final" if (root / "final").is_dir() else root

    references = {
        path.stem.lower(): path
        for path in sorted(final_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not references:
        return []

    brackets = group_brackets(list_supported_files(raw_dir))
    pairs: list[tuple[Bracket, Path]] = []

    for bracket in brackets:
        # El export puede llamarse como el bracket o como cualquiera de sus tomas.
        candidates = [bracket.name.lower(), *(p.stem.lower() for p in bracket.paths)]
        for candidate in candidates:
            if candidate in references:
                pairs.append((bracket, references[candidate]))
                break

    return pairs


def _load_reference(path: Path) -> np.ndarray:
    data = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if data is None:
        raise ValueError(f"no se pudo abrir la referencia {path}")
    rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return resize_long_side(rgb, COMPARE_LONG_SIDE)


# --------------------------------------------------------------------------
# Ajuste
# --------------------------------------------------------------------------


def _calibration_config(cfg: Config, learned: dict) -> Config:
    """Configuracion de calibracion: el pipeline real menos lo que estorba.

    Se desactiva solo lo que impide comparar (la geometria cambia el encuadre y
    tu export lleva tu propio recorte) o lo que no aporta y cuesta tiempo. Todo
    lo demas queda activo a proposito: lo que se esta midiendo es el error del
    pipeline completo, no el de una version simplificada de el.
    """
    overrides: dict = {
        "batch": {"consistency": 0.0},
        "cleanup": {"enabled": False},
        "geometry": {"verticals": {"enabled": False}, "straighten": {"enabled": False}},
        "output": {"max_long_side": COMPARE_LONG_SIDE},
    }
    return cfg.with_overrides(deep_merge(overrides, learned))


def _render(bracket: Bracket, cfg: Config) -> np.ndarray:
    from .pipeline import render_scene

    output = render_scene(bracket, cfg, analysis=True)
    return resize_long_side(output.image, COMPARE_LONG_SIDE)


def fit_curve(neutral: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Curva de tono que lleva la luminancia del render a la de tu export.

    Se ajusta por igualacion de histogramas y no comparando pixel a pixel: tu
    export lleva tu recorte, asi que las dos imagenes no coinciden pixel con
    pixel, pero la distribucion de tonos si es comparable.
    """
    source = np.sort(luminance(neutral).reshape(-1))
    target = np.sort(luminance(reference).reshape(-1))

    positions = np.linspace(0.0, 1.0, LUT_SIZE, dtype=np.float32)
    # Para cada nivel de entrada: en que percentil cae, y que valor ocupa ese
    # mismo percentil en la referencia.
    quantiles = np.searchsorted(source, positions) / max(len(source), 1)
    curve = np.interp(quantiles, np.linspace(0.0, 1.0, len(target)), target)

    # La igualacion de histogramas es ruidosa en los extremos, donde hay pocos
    # pixeles. Un suavizado ligero y la monotonia forzada evitan escalones.
    kernel = np.ones(9, dtype=np.float32) / 9.0
    curve = np.convolve(np.pad(curve, 4, mode="edge"), kernel, mode="valid")
    return np.clip(np.maximum.accumulate(curve), 0.0, 1.0).astype(np.float32)


def distribution_error(rendered: np.ndarray, reference: np.ndarray) -> float:
    """Distancia entre dos imagenes comparando sus distribuciones por canal.

    No se comparan pixel a pixel porque tu export lleva tu propio recorte y las
    dos imagenes no coinciden. Ordenar los valores de cada canal quita de la
    ecuacion donde esta cada cosa y deja solo como estan tratados el tono y el
    color, que es lo que se esta calibrando.
    """
    a = np.sort(rendered.reshape(-1, 3), axis=0)
    b = np.sort(reference.reshape(-1, 3), axis=0)
    if len(a) != len(b):
        positions = np.linspace(0.0, 1.0, min(len(a), len(b)), dtype=np.float32)
        a = np.stack([np.interp(positions, np.linspace(0, 1, len(a)), a[:, c]) for c in range(3)], 1)
        b = np.stack([np.interp(positions, np.linspace(0, 1, len(b)), b[:, c]) for c in range(3)], 1)
    return float(np.abs(a - b).mean())


def _chroma_strength(image: np.ndarray) -> float:
    gray = luminance(image)[..., None]
    return float(np.abs(image - gray).mean() / max(gray.mean(), EPS))


def calibrate(root: Path, cfg: Config) -> CalibrationResult:
    """Mide tu look sobre los pares y devuelve los parametros aprendidos."""
    pairs = find_pairs(Path(root))
    result = CalibrationResult()

    if not pairs:
        result.notes.append(
            "no se encontro ningun par: revisa que 'raw/' tenga los brackets y "
            "'final/' los JPEG exportados con el mismo nombre"
        )
        return result

    references = {}
    for bracket, reference_path in pairs:
        try:
            references[bracket.name] = _load_reference(reference_path)
        except ValueError as exc:
            result.notes.append(str(exc))

    if not references:
        return result

    # --- Paso 1: cuanto de clara dejas una foto ------------------------------
    midtones = [
        interior_level(luminance(reference)) for reference in references.values()
    ]
    result.target_midtone = float(np.clip(np.median(midtones), 0.2, 0.75))

    # --- Paso 2: bucle de refinado -------------------------------------------
    #
    # Curva, calidez y saturacion no se pueden medir por separado: un tinte
    # calido anade croma, y subir la saturacion desplaza las medias de canal.
    # En vez de modelar ese acoplamiento, se cierra el bucle -- se revela con
    # los parametros actuales, se mide el error que queda y se corrige -- y las
    # tres estimaciones convergen juntas en unas pocas vueltas.
    curve = build_tone_curve(
        contrast=float(cfg.grade.contrast),
        black_point=float(cfg.grade.black_point),
        white_point=float(cfg.grade.white_point),
        size=LUT_SIZE,
    )
    warmth = float(cfg.white_balance.keep_warmth)
    saturation = float(cfg.grade.saturation)
    best = (curve.copy(), warmth, saturation)
    best_error = float("inf")

    for iteration in range(REFINE_ITERATIONS):
        learned = {
            "tone": {"target_midtone": result.target_midtone},
            "white_balance": {"keep_warmth": warmth},
            "grade": {"saturation": saturation, "curve_lut": [float(v) for v in curve]},
        }
        step_cfg = _calibration_config(cfg, learned)

        curves: list[np.ndarray] = []
        warmth_deltas: list[float] = []
        saturation_ratios: list[float] = []
        errors: list[float] = []
        used = 0

        for bracket, _ in pairs:
            reference = references.get(bracket.name)
            if reference is None:
                continue
            try:
                rendered = _render(bracket, step_cfg)
            except Exception as exc:  # noqa: BLE001 - un bracket roto no invalida el resto
                if iteration == 0:
                    result.notes.append(f"{bracket.name}: no se pudo revelar ({exc})")
                continue
            used += 1
            errors.append(distribution_error(rendered, reference))

            # La curva correctora se compone sobre la que ya estaba puesta:
            # total(x) = correccion(curva(x)).
            curves.append(apply_lut(curve, fit_curve(rendered, reference)))

            rendered_signature = image_signature(rendered)
            reference_signature = image_signature(reference)
            red = reference_signature["r_over_g"] / max(rendered_signature["r_over_g"], EPS)
            blue = reference_signature["b_over_g"] / max(rendered_signature["b_over_g"], EPS)
            # keep_warmth aplica ganancias [1+0.18k, 1, 1-0.16k], pero lo hace en
            # espacio lineal mientras que el desvio se mide en pantalla. La gamma
            # sRGB comprime la proporcion (una ganancia lineal g se ve como
            # g^(1/2.4)), de modo que sin compensarla cada vuelta corregiria poco
            # mas de un tercio del error y tres vueltas no bastarian.
            warmth_deltas.append(
                GAMMA * float(np.mean([(red - 1.0) / 0.18, (1.0 - blue) / 0.16]))
            )
            saturation_ratios.append(
                _chroma_strength(reference) / max(_chroma_strength(rendered), EPS)
            )

        if not curves:
            result.notes.append("ningun par se pudo procesar")
            return result

        result.pairs_used = used

        # Se guarda el mejor estado medido, no el ultimo. Calidez y saturacion
        # estan acopladas y el bucle puede pasarse de frenada en una vuelta; con
        # esto, calibrar nunca deja el look peor de como estaba.
        error = float(np.mean(errors))
        if error < best_error:
            best_error = error
            best = (curve.copy(), warmth, saturation)

        # Paso amortiguado: la correccion completa de cada parametro ignora que
        # el otro se mueve a la vez, y a plena ganancia los dos oscilan.
        curve_target = np.clip(
            np.maximum.accumulate(np.median(np.stack(curves), axis=0)), 0.0, 1.0
        )
        curve = ((1.0 - DAMPING) * curve + DAMPING * curve_target).astype(np.float32)
        warmth = float(
            np.clip(warmth + DAMPING * np.median(warmth_deltas), 0.0, WARMTH_LIMIT)
        )
        saturation = float(
            np.clip(saturation * (1.0 + DAMPING * (np.median(saturation_ratios) - 1.0)), 0.7, 1.8)
        )

    result.curve, result.keep_warmth, result.saturation = best
    result.error = best_error

    # Se comprueba el resultado que se guarda, no el ultimo valor del bucle:
    # una vuelta intermedia puede tocar el tope y luego volver a rango.
    for label, value, limits in (
        ("keep_warmth", result.keep_warmth, (0.0, WARMTH_LIMIT)),
        ("saturation", result.saturation, (0.7, 1.8)),
    ):
        if value in limits:
            result.notes.append(
                f"{label} ha quedado en su tope ({value}): tus exports se separan "
                "mucho del render base. Revisa que las referencias correspondan "
                "de verdad a esos RAW."
            )

    if result.pairs_used < 10:
        result.notes.append(
            f"solo {result.pairs_used} pares utiles: con menos de 10 la curva "
            "aprendida es inestable, mejor anadir mas escenas variadas"
        )

    return result


def write_learned(result: CalibrationResult, path: Path) -> Path:
    """Guarda lo aprendido donde `load_config` lo recoge automaticamente."""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    merged = deep_merge(existing, result.to_overrides())
    header = (
        "# Generado por `hdrpipe calibrate` a partir de tus pares antes/despues.\n"
        f"# Pares utilizados: {result.pairs_used}\n"
        "# Se aplica encima de config/default.yaml. Borra este archivo para\n"
        "# volver a los valores de fabrica.\n"
    )
    path.write_text(
        header + yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return path
