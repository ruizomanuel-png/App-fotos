"""Interfaz de linea de comandos.

    hdrpipe run entrada/ salida/     procesa una carpeta
    hdrpipe calibrate calibracion/   aprende tu look de los pares antes/despues
    hdrpipe serve                    levanta la app web en localhost
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import LEARNED_CONFIG_PATH, load_config


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML de configuracion alternativo"
    )
    parser.add_argument(
        "--sin-calibrar",
        action="store_true",
        help="ignora config/learned.yaml y usa los valores de fabrica",
    )


def _load(args: argparse.Namespace):
    return load_config(
        path=args.config,
        learned=None if getattr(args, "sin_calibrar", False) else LEARNED_CONFIG_PATH,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_batch, summarize

    cfg = _load(args)
    if args.rapido:
        cfg = cfg.with_overrides({"raw": {"working_max_side": 1600}})

    def progress(message: str, done: int, total: int) -> None:
        print(f"[{done:>3}/{total}] {message}", flush=True)

    result = run_batch(
        args.entrada, args.salida, cfg, progress=progress, limit=args.limite
    )
    print()
    print(summarize(result))
    print(f"\nInforme detallado en {Path(args.salida) / 'informe.json'}")
    return 0 if not result.errored else 1


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibrate import calibrate, write_learned

    cfg = load_config(path=args.config, learned=None)
    result = calibrate(args.carpeta, cfg)

    for note in result.notes:
        print(f"aviso: {note}", file=sys.stderr)

    if result.pairs_used == 0:
        print("no se ha podido calibrar nada", file=sys.stderr)
        return 1

    path = write_learned(result, LEARNED_CONFIG_PATH)
    print(f"Calibrado con {result.pairs_used} pares.")
    print(f"  luminancia media objetivo : {result.target_midtone:.3f}")
    print(f"  calidez conservada        : {result.keep_warmth:.3f}")
    print(f"  saturacion                : {result.saturation:.3f}")
    print(f"  curva de tono             : {'aprendida' if result.curve is not None else 'no'}")
    print(f"  error residual            : {result.error:.4f}")
    print(f"\nGuardado en {path}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Abre http://127.0.0.1:{args.puerto} en el navegador")
    uvicorn.run("webapp.main:app", host=args.host, port=args.puerto, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hdrpipe", description="Edicion automatica de fotografia HDR inmobiliaria"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log detallado")
    subparsers = parser.add_subparsers(dest="comando", required=True)

    run = subparsers.add_parser("run", help="procesa una carpeta de brackets")
    run.add_argument("entrada", type=Path)
    run.add_argument("salida", type=Path)
    run.add_argument("--limite", type=int, default=None, help="procesa solo N escenas")
    run.add_argument(
        "--rapido",
        action="store_true",
        help="trabaja a 1600 px de lado largo, para revisar el look sin esperar",
    )
    _add_config_flags(run)
    run.set_defaults(func=_cmd_run)

    calibrate_parser = subparsers.add_parser(
        "calibrate", help="aprende tu look de los pares antes/despues"
    )
    calibrate_parser.add_argument("carpeta", type=Path, help="carpeta con raw/ y final/")
    calibrate_parser.add_argument("--config", type=Path, default=None)
    calibrate_parser.set_defaults(func=_cmd_calibrate)

    serve = subparsers.add_parser("serve", help="levanta la app web")
    serve.add_argument("--puerto", type=int, default=8000)
    serve.add_argument("--host", default="127.0.0.1")
    serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
