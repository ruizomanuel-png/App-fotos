"""Orquestacion: de una carpeta de RAW a una carpeta de JPEG.

El lote se procesa en dos pasadas:

1. **Analisis**, a baja resolucion. Solo mide tono y color de cada escena. Es
   rapida y sirve para calcular el consenso del lote.
2. **Render**, a resolucion completa, con la correccion de consistencia ya
   incorporada.

Hacerlo asi evita tener cuarenta imagenes de 40 megapixeles en memoria a la vez
y evita comprimir el JPEG dos veces.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import batch as batch_mod
from .align import align_frames
from .cleanup import remove_objects
from .color import grade, image_signature, white_balance
from .config import Config
from .detail import enhance
from .geometry import correct_geometry
from .grouping import Bracket, group_brackets, list_supported_files
from .imageops import resize_long_side, to_uint8
from .merge import merge_frames
from .quality import QualityReport, inspect
from .raw import load_bracket
from .tone import estimate_window_mask, map_tone

log = logging.getLogger(__name__)

# Lado largo de la pasada de analisis. Suficiente para medir tono y color, y
# unas cincuenta veces mas rapido que trabajar a resolucion completa.
ANALYSIS_LONG_SIDE = 900

ProgressCallback = Callable[[str, int, int], None]


@dataclass
class SceneResult:
    name: str
    sources: list[str]
    output: str | None = None
    delivered: bool = True
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass
class BatchResult:
    scenes: list[SceneResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def delivered(self) -> list[SceneResult]:
        return [s for s in self.scenes if s.delivered and not s.failed]

    @property
    def review(self) -> list[SceneResult]:
        return [s for s in self.scenes if not s.delivered and not s.failed]

    @property
    def errored(self) -> list[SceneResult]:
        return [s for s in self.scenes if s.failed]

    def to_dict(self) -> dict:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "total": len(self.scenes),
            "entregadas": len(self.delivered),
            "a_revisar": len(self.review),
            "con_error": len(self.errored),
            "escenas": [
                {
                    "nombre": s.name,
                    "origen": s.sources,
                    "salida": s.output,
                    "entregada": s.delivered,
                    "motivos": s.issues,
                    "notas": s.notes,
                    "metricas": {k: round(float(v), 4) for k, v in s.metrics.items()},
                    "error": s.error,
                }
                for s in self.scenes
            ],
        }


# --------------------------------------------------------------------------
# Render de una escena
# --------------------------------------------------------------------------


@dataclass
class RenderOutput:
    image: np.ndarray
    signature: dict[str, float]
    notes: list[str] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    bracket_size: int = 0


def render_scene(
    bracket: Bracket,
    cfg: Config,
    *,
    analysis: bool = False,
    correction: batch_mod.Correction | None = None,
) -> RenderOutput:
    """Procesa un bracket completo.

    En modo `analysis` se trabaja a baja resolucion y se saltan las etapas que
    no afectan al tono global (geometria, borrado de objetos, enfoque), que son
    justamente las mas lentas.
    """
    notes: list[str] = list(bracket.warnings)
    blocking: list[str] = []
    metrics: dict = {}

    max_side = ANALYSIS_LONG_SIDE if analysis else cfg.raw.working_max_side
    frames = load_bracket(
        bracket.paths,
        use_camera_wb=bool(cfg.raw.use_camera_wb),
        demosaic=str(cfg.raw.demosaic),
        max_long_side=max_side,
    )

    frames, align_warnings = align_frames(
        frames,
        enabled=bool(cfg.align.enabled) and not analysis,
        max_shift_px=float(cfg.align.max_shift_px),
        estimate_max_side=int(cfg.align.estimate_max_side),
    )
    notes.extend(align_warnings)
    # Una toma sin alinear deja bordes dobles en los marcos de ventana.
    blocking.extend(w for w in align_warnings if "sin alinear" in w or "no se pudo" in w)

    merged = merge_frames(
        frames,
        mode=str(cfg.merge.mode),
        low_cut=float(cfg.merge.low_cut),
        high_cut=float(cfg.merge.high_cut),
    )
    notes.extend(merged.warnings)
    metrics["ev_spread"] = merged.ev_spread

    window_hint = estimate_window_mask(merged.image, cfg)
    linear, _ = white_balance(merged.image, window_hint, cfg)

    toned = map_tone(linear, cfg)
    metrics["window_ev"] = toned.applied_window_ev
    metrics["shadow_ev"] = toned.applied_shadow_ev
    metrics["window_area"] = toned.stats["window_area"]

    display = toned.image

    if not analysis:
        cleaned = remove_objects(display, cfg)
        display = cleaned.image
        notes.extend(cleaned.warnings)
        blocking.extend(cleaned.blocking)
        if cleaned.removed:
            notes.append("borrado automatico: " + ", ".join(sorted(set(cleaned.removed))))

        geometry = correct_geometry(
            display,
            cfg,
            lens=frames[0].lens,
            focal_35mm=frames[0].focal_35mm,
        )
        display = geometry.image
        notes.extend(geometry.warnings)
        metrics["geometry_degrees"] = geometry.applied_degrees
        metrics["crop_area_loss"] = geometry.area_loss

    display = grade(display, cfg)
    signature = image_signature(display)

    if correction is not None:
        display = batch_mod.apply_correction(display, correction)

    if not analysis:
        display = resize_long_side(display, cfg.output.max_long_side)
        # El enfoque va al final, ya al tamano de entrega: enfocar antes y
        # reducir despues desperdicia el ajuste.
        display = enhance(display, cfg)

    return RenderOutput(
        image=display,
        signature=signature,
        notes=notes,
        blocking=blocking,
        metrics=metrics,
        bracket_size=bracket.size,
    )


# --------------------------------------------------------------------------
# Guardado
# --------------------------------------------------------------------------


def save_image(display: np.ndarray, path: Path, cfg: Config) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(to_uint8(display), mode="RGB")

    if str(cfg.output.format).lower() in {"jpeg", "jpg"}:
        image.save(
            path,
            format="JPEG",
            quality=int(cfg.output.quality),
            subsampling=str(cfg.output.subsampling),
            optimize=True,
            progressive=True,
        )
    else:
        image.save(path)


# --------------------------------------------------------------------------
# Lote completo
# --------------------------------------------------------------------------


def run_batch(
    input_dir: Path,
    output_dir: Path,
    cfg: Config,
    *,
    progress: ProgressCallback | None = None,
    limit: int | None = None,
) -> BatchResult:
    """Procesa una carpeta entera. Devuelve el informe del lote."""
    started = time.monotonic()
    input_dir, output_dir = Path(input_dir), Path(output_dir)

    files = list_supported_files(input_dir)
    brackets = group_brackets(
        files,
        bracket_size=int(cfg.grouping.bracket_size),
        max_seconds_between_shots=float(cfg.grouping.max_seconds_between_shots),
    )
    if limit:
        brackets = brackets[:limit]

    result = BatchResult()
    if not brackets:
        result.elapsed_seconds = time.monotonic() - started
        return result

    review_dir = output_dir / "revisar"
    consistency = float(cfg.batch.consistency)

    # --- Pasada 1: analisis --------------------------------------------------
    profile = batch_mod.BatchProfile(consistency=consistency)
    signatures: dict[str, dict[str, float]] = {}

    if consistency > 0 and len(brackets) > 1:
        for index, bracket in enumerate(brackets, start=1):
            if progress:
                progress(f"analizando {bracket.name}", index, len(brackets) * 2)
            try:
                signatures[bracket.name] = render_scene(bracket, cfg, analysis=True).signature
            except Exception as exc:  # noqa: BLE001 - una escena rota no para el lote
                log.warning("no se pudo analizar %s: %s", bracket.name, exc)
        profile = batch_mod.build_profile(list(signatures.values()), consistency)

    # --- Pasada 2: render ----------------------------------------------------
    offset = len(brackets) if signatures else 0
    total = len(brackets) * 2 if signatures else len(brackets)

    for index, bracket in enumerate(brackets, start=1):
        if progress:
            progress(f"revelando {bracket.name}", offset + index, total)

        scene = SceneResult(name=bracket.name, sources=[p.name for p in bracket.paths])
        try:
            correction = (
                batch_mod.correction_for(signatures[bracket.name], profile)
                if bracket.name in signatures
                else None
            )
            rendered = render_scene(bracket, cfg, correction=correction)

            report: QualityReport = inspect(
                rendered.image,
                cfg,
                bracket_size=rendered.bracket_size,
                blocking=rendered.blocking,
            )

            destination = (output_dir if report.passed else review_dir) / f"{bracket.name}.jpg"
            save_image(rendered.image, destination, cfg)

            scene.output = str(destination.relative_to(output_dir))
            scene.delivered = report.passed
            scene.issues = report.issues
            scene.notes = rendered.notes
            scene.metrics = {**rendered.metrics, **report.metrics}
            if correction is not None and not correction.is_identity:
                scene.metrics["batch_exposure_gain"] = correction.exposure

        except Exception as exc:  # noqa: BLE001 - se registra y se sigue con el lote
            log.exception("fallo procesando %s", bracket.name)
            scene.error = f"{type(exc).__name__}: {exc}"
            scene.delivered = False

        result.scenes.append(scene)

    result.elapsed_seconds = time.monotonic() - started
    write_report(result, output_dir)
    return result


def write_report(result: BatchResult, output_dir: Path) -> Path:
    """Deja un informe JSON junto a las fotos, con lo que se hizo en cada una."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "informe.json"
    path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def summarize(result: BatchResult) -> str:
    """Resumen de una linea por escena, para la consola."""
    lines = [
        f"{len(result.delivered)} entregadas, {len(result.review)} a revisar, "
        f"{len(result.errored)} con error, en {result.elapsed_seconds:.0f} s"
    ]
    for scene in result.scenes:
        if scene.failed:
            lines.append(f"  ERROR   {scene.name}: {scene.error}")
        elif not scene.delivered:
            lines.append(f"  REVISAR {scene.name}: {'; '.join(scene.issues)}")
    return "\n".join(lines)
