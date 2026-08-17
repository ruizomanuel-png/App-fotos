"""Pruebas del pipeline sobre escenas sinteticas.

Las escenas de `scenes.py` reproducen el caso dificil de siempre: interior en
penumbra, rincon oscuro y ventana cinco paradas por encima. Es donde se rompen
las cosas, asi que es donde se comprueban.
"""

from __future__ import annotations

import numpy as np
import pytest

from hdrpipe.color import build_tone_curve, estimate_illuminant, white_balance
from hdrpipe.config import Config, deep_merge, load_config
from hdrpipe.geometry import correct_geometry, estimate_vertical_vp, reframe_homography
from hdrpipe.grouping import group_brackets, list_supported_files
from hdrpipe.imageops import guided_filter, luminance, srgb_decode, srgb_encode
from hdrpipe.merge import merge_frames
from hdrpipe.pipeline import render_scene, run_batch
from hdrpipe.raw import Frame
from hdrpipe.tone import map_tone

from .scenes import (
    BASE_SHUTTER,
    SENSOR_GAIN,
    SHUTTER_SPEEDS,
    interior_scene,
    write_batch,
    write_bracket,
)


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(learned=None)


def make_frames(scene: np.ndarray, seed: int = 0) -> list[Frame]:
    """Convierte una escena en las tres tomas del bracket, con EXIF completo."""
    from pathlib import Path

    rng = np.random.default_rng(seed)
    frames = []
    for shutter in SHUTTER_SPEEDS:
        exposed = scene * (shutter / BASE_SHUTTER) * SENSOR_GAIN
        exposed = np.clip(exposed + rng.normal(0, 0.0015, exposed.shape), 0, 1)
        frames.append(
            Frame(
                path=Path(f"toma_{shutter}.arw"),
                image=exposed.astype(np.float32),
                exposure_time=shutter,
                f_number=8.0,
                iso=100,
            )
        )
    return frames


# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------


def test_config_acceso_por_puntos(cfg):
    assert isinstance(cfg.tone.window_pull.ev, (int, float))
    with pytest.raises(AttributeError):
        _ = cfg.tone.no_existe


def test_config_es_inmutable(cfg):
    with pytest.raises(AttributeError):
        cfg.tone.target_midtone = 0.9


def test_overrides_no_mutan_el_original(cfg):
    original = cfg.tone.target_midtone
    modificada = cfg.with_overrides({"tone": {"target_midtone": 0.9}})
    assert modificada.tone.target_midtone == 0.9
    assert cfg.tone.target_midtone == original
    # El resto de la seccion sobrevive a la fusion.
    assert modificada.tone.window_pull.ev == cfg.tone.window_pull.ev


def test_deep_merge_es_recursivo():
    resultado = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}})
    assert resultado == {"a": {"b": 1, "c": 3}}


# --------------------------------------------------------------------------
# Operaciones basicas
# --------------------------------------------------------------------------


def test_srgb_ida_y_vuelta():
    valores = np.linspace(0, 1, 128, dtype=np.float32)
    assert np.allclose(srgb_decode(srgb_encode(valores)), valores, atol=1e-5)


def test_guided_filter_conserva_los_bordes():
    """Un filtro guiado suaviza dentro de cada zona pero respeta el escalon."""
    guide = np.zeros((200, 200), np.float32)
    guide[:, 100:] = 1.0
    ruidosa = guide + np.random.default_rng(0).normal(0, 0.1, guide.shape).astype(np.float32)

    filtrada = guided_filter(guide, ruidosa, radius=8, eps=1e-4)

    assert filtrada[:, :90].std() < ruidosa[:, :90].std() / 2   # suaviza
    assert filtrada[:, 110:].mean() - filtrada[:, :90].mean() > 0.9   # mantiene el escalon


def test_curva_de_tono_monotona():
    for contraste in (0.0, 0.2, 0.5, 1.0):
        curva = build_tone_curve(contrast=contraste, black_point=0.01, white_point=0.99)
        assert np.all(np.diff(curva) >= -1e-6), f"no monotona con contraste {contraste}"


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def test_la_fusion_recupera_la_ventana_quemada():
    """La toma central quema la ventana; la fusion debe conservar ese detalle."""
    escena = interior_scene()
    frames = make_frames(escena)

    central = frames[1].image
    assert (central[90:260, 380:550].max(axis=2) >= 0.985).mean() > 0.5

    fusionada = merge_frames(frames, mode="radiance")

    assert fusionada.method == "radiance"
    assert fusionada.ev_spread == pytest.approx(4.0, abs=0.1)
    # La radiancia de la ventana queda muy por encima del techo de una sola toma.
    assert fusionada.image[90:260, 380:550].max() > 1.5


def test_la_fusion_deduce_la_exposicion_sin_exif():
    """Sin EXIF, las escalas se estiman de los pixeles y el resultado coincide."""
    frames = make_frames(interior_scene())
    con_exif = merge_frames(frames, mode="radiance")

    for frame in frames:
        frame.exposure_time = None
        frame.iso = None
    sin_exif = merge_frames(frames, mode="radiance")

    assert sin_exif.warnings
    ratio = np.median(sin_exif.image) / np.median(con_exif.image)
    assert 0.8 < ratio < 1.25


def test_sin_separacion_de_exposicion_cae_a_mertens():
    escena = interior_scene()
    frames = make_frames(escena)[:1] * 3
    for frame in frames:
        frame.exposure_time = 1 / 50
    assert merge_frames(frames, mode="radiance").method == "fusion"


# --------------------------------------------------------------------------
# Tono
# --------------------------------------------------------------------------


def test_el_window_pull_saca_detalle_de_la_ventana(cfg):
    fusionada = merge_frames(make_frames(interior_scene()), mode="radiance")
    resultado = map_tone(fusionada.image, cfg)

    assert resultado.applied_window_ev > 1.0
    # La mascara encuentra la ventana, que ocupa cerca del 16% del encuadre.
    assert 0.08 < resultado.stats["window_area"] < 0.30

    ventana = resultado.image[90:260, 380:550]
    assert float((ventana.max(axis=2) >= 0.995).mean()) < 0.05     # ya no esta quemada
    assert ventana.mean(axis=2).std() > 0.02                       # y tiene contraste


def test_la_mascara_de_ventana_no_depende_del_area(cfg):
    """El umbral va en paradas sobre el interior, no en percentiles.

    Con percentiles, una ventana que ocupa un tercio del encuadre quedaria
    parcialmente fuera de su propia mascara.
    """
    for ancho in (0.10, 0.35):
        escena = interior_scene()
        escena[60:260, 360:560] = escena[100, 100]        # se borra la ventana original
        derecha = int(600 * (1.0 - ancho))
        escena[80:300, derecha:] = [0.55, 0.65, 0.95]     # ventana del ancho pedido

        resultado = map_tone(merge_frames(make_frames(escena), mode="radiance").image, cfg)
        assert resultado.applied_window_ev > 1.0, f"sin correccion con ancho {ancho}"


def test_las_altas_luces_se_desaturan_en_vez_de_recortarse(cfg):
    """Un cielo muy azul debe tender a blanco, no saturar el canal azul."""
    fusionada = merge_frames(make_frames(interior_scene()), mode="radiance")
    resultado = map_tone(fusionada.image, cfg)
    cielo = resultado.image[70:95, 370:550].reshape(-1, 3)
    assert float((cielo.max(axis=1) >= 0.998).mean()) < 0.05


# --------------------------------------------------------------------------
# Color
# --------------------------------------------------------------------------


def test_el_balance_de_blancos_neutraliza_la_dominante(cfg):
    gris = np.full((200, 300, 3), 0.18, np.float32)
    dominante = gris * np.array([1.35, 1.0, 0.62], np.float32)   # tungsteno

    corregida, _ = white_balance(dominante, None, cfg)

    medias = corregida.reshape(-1, 3).mean(axis=0)
    desvio = float(abs(medias[0] / medias[1] - 1.0) + abs(medias[2] / medias[1] - 1.0))
    original = float(abs(1.35 - 1.0) + abs(0.62 - 1.0))
    assert desvio < original / 3


def test_el_iluminante_se_normaliza_a_verde():
    imagen = np.full((100, 100, 3), 0.2, np.float32) * np.array([1.2, 1.0, 0.8], np.float32)
    assert estimate_illuminant(imagen)[1] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------


def _inclinar(imagen: np.ndarray, vp_y: float) -> np.ndarray:
    import cv2

    alto, ancho = imagen.shape[:2]
    K = np.array([[1.2 * ancho, 0, ancho / 2], [0, 1.2 * ancho, alto / 2], [0, 0, 1]], float)
    direccion = np.linalg.inv(K) @ np.array([ancho / 2, vp_y, 1.0])
    direccion /= np.linalg.norm(direccion)
    objetivo = np.array([0.0, -1.0, 0.0])
    if direccion @ objetivo < 0:
        direccion = -direccion
    eje = np.cross(direccion, objetivo)
    seno = np.linalg.norm(eje)
    R, _ = cv2.Rodrigues(eje / seno * np.arctan2(seno, direccion @ objetivo))
    H = np.linalg.inv(K @ R @ np.linalg.inv(K))
    return cv2.warpPerspective(imagen, H / H[2, 2], (ancho, alto), borderMode=cv2.BORDER_REPLICATE)


def test_se_localiza_el_punto_de_fuga_vertical():
    from hdrpipe.geometry import detect_segments

    recta = srgb_encode(np.clip(interior_scene(600, 900) * 4.0, 0, 1))
    inclinada = _inclinar(recta, -4500)

    vp, lineas = estimate_vertical_vp(detect_segments(inclinada), inclinada.shape[:2])

    assert vp is not None and lineas >= 6
    assert vp[1] == pytest.approx(-4500, rel=0.05)


def test_se_enderezan_las_verticales_sin_recortar_de_mas(cfg):
    from hdrpipe.geometry import _segment_angles, _segment_lengths, detect_segments

    def desviacion(imagen):
        segmentos = detect_segments(imagen)
        angulos, largos = _segment_angles(segmentos), _segment_lengths(segmentos)
        interes = (angulos < 30) & (largos > 0.06 * imagen.shape[0])
        return float(np.average(angulos[interes], weights=largos[interes]))

    recta = srgb_encode(np.clip(interior_scene(600, 900) * 4.0, 0, 1))
    inclinada = _inclinar(recta, -9000)          # unos 6.6 grados, lo habitual
    resultado = correct_geometry(inclinada, cfg)

    assert resultado.method == "verticals"
    assert desviacion(resultado.image) < desviacion(inclinada) / 2
    # Reencuadrar dentro del cuadrilatero corregido, y no contra el lienzo de
    # partida, mantiene la perdida proporcionada a la correccion. Contra el
    # lienzo de partida, esta misma escena costaba mas del 30%.
    assert resultado.area_loss < 0.08


def test_una_escena_ya_recta_no_se_toca(cfg):
    recta = srgb_encode(np.clip(interior_scene(600, 900) * 4.0, 0, 1))
    resultado = correct_geometry(recta, cfg)
    assert resultado.applied_degrees == 0.0
    assert np.array_equal(resultado.image, recta)


def test_el_reencuadre_aprovecha_casi_todo():
    """Una rotacion pequena no puede costar un tercio de la foto."""
    import cv2

    alto, ancho = 600, 900
    rotacion = cv2.getRotationMatrix2D((ancho / 2, alto / 2), 2.0, 1.0)
    H = np.vstack([rotacion, [0, 0, 1]])
    _, perdida = reframe_homography(H, (alto, ancho))
    assert perdida < 0.12


# --------------------------------------------------------------------------
# Agrupacion
# --------------------------------------------------------------------------


def test_los_brackets_se_agrupan_de_tres_en_tres(tmp_path):
    write_batch(tmp_path, count=4)
    brackets = group_brackets(list_supported_files(tmp_path))

    assert len(brackets) == 4
    assert all(bracket.size == 3 for bracket in brackets)
    assert not any(bracket.warnings for bracket in brackets)


def test_el_nombre_de_escena_pierde_el_indice_de_toma(tmp_path):
    write_bracket(tmp_path, "salon", interior_scene())
    (bracket,) = group_brackets(list_supported_files(tmp_path))
    assert bracket.name == "salon"


def test_un_bracket_incompleto_queda_avisado(tmp_path):
    rutas = write_bracket(tmp_path, "salon", interior_scene())
    rutas[-1].unlink()
    (bracket,) = group_brackets(list_supported_files(tmp_path))
    assert bracket.size == 2 and bracket.warnings


# --------------------------------------------------------------------------
# Lote completo
# --------------------------------------------------------------------------


def test_el_lote_completo_entrega_jpeg(tmp_path, cfg):
    origen, destino = tmp_path / "entrada", tmp_path / "salida"
    write_batch(origen, count=3)

    resultado = run_batch(origen, destino, cfg)

    assert len(resultado.scenes) == 3
    assert not resultado.errored
    assert len(resultado.delivered) == 3
    assert sorted(p.name for p in destino.glob("*.jpg")) == [
        "escena01.jpg", "escena02.jpg", "escena03.jpg"
    ]
    assert (destino / "informe.json").is_file()


def test_la_consistencia_iguala_el_tono_del_lote(tmp_path, cfg):
    """Escenas con brillos distintos deben acabar pareciendose entre si."""
    origen = tmp_path / "entrada"
    write_batch(origen, count=4)
    brackets = group_brackets(list_supported_files(origen))

    def dispersion(configuracion):
        medias = [
            float(luminance(render_scene(b, configuracion, analysis=True).image).mean())
            for b in brackets
        ]
        return float(np.std(medias))

    sueltas = dispersion(cfg.with_overrides({"batch": {"consistency": 0.0}}))

    salida = tmp_path / "salida"
    resultado = run_batch(origen, salida, cfg)
    assert len(resultado.delivered) == 4

    import cv2

    medias = [
        float(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).mean() / 255.0)
        for p in sorted(salida.glob("*.jpg"))
    ]
    assert float(np.std(medias)) < sueltas


def test_una_foto_desenfocada_va_a_revisar(tmp_path, cfg):
    import cv2

    origen, destino = tmp_path / "entrada", tmp_path / "salida"
    write_batch(origen, count=2)
    # Se emborrona una de las escenas hasta hacerla inservible.
    for ruta in sorted(origen.glob("escena02_*.tif")):
        cv2.imwrite(str(ruta), cv2.GaussianBlur(cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED), (0, 0), 9))

    resultado = run_batch(origen, destino, cfg)

    revisar = {escena.name for escena in resultado.review}
    assert "escena02" in revisar
    assert (destino / "revisar" / "escena02.jpg").is_file()
    assert not (destino / "escena02.jpg").exists()


def test_una_escena_rota_no_tumba_el_lote(tmp_path, cfg):
    origen, destino = tmp_path / "entrada", tmp_path / "salida"
    write_batch(origen, count=2)
    (origen / "escena01_2.tif").write_bytes(b"esto no es una imagen")

    resultado = run_batch(origen, destino, cfg)

    assert len(resultado.errored) == 1
    assert len(resultado.delivered) == 1   # la otra escena sale adelante
