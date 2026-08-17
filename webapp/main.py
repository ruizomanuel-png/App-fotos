"""App web local.

Corre en tu propio ordenador y se abre en el navegador. No hay servidor, ni
cuentas, ni factura: es una interfaz comoda encima del pipeline.

    hdrpipe serve      ->  http://127.0.0.1:8000
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from hdrpipe.config import load_config
from hdrpipe.grouping import list_supported_files
from hdrpipe.raw import SUPPORTED_EXTENSIONS

from .jobs import JobStore, purge_expired

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
PURGE_INTERVAL_SECONDS = 3600

store = JobStore()


def _retention_loop() -> None:
    """Aplica la caducidad de una semana, ahora y cada hora."""
    while True:
        try:
            days = int(load_config().retention.days)
            removed = purge_expired(store, days)
            if removed:
                log.info("%d trabajos caducados eliminados", removed)
        except Exception:  # noqa: BLE001 - la limpieza nunca debe tumbar la app
            log.exception("fallo la limpieza de trabajos caducados")
        time.sleep(PURGE_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_retention_loop, daemon=True).start()
    yield


app = FastAPI(title="Editor HDR inmobiliario", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/jobs")
async def list_jobs() -> JSONResponse:
    return JSONResponse([job.to_dict() for job in store.list()])


@app.post("/api/jobs/upload")
async def create_from_upload(files: list[UploadFile], name: str = Form("lote")) -> JSONResponse:
    """Crea un trabajo copiando los archivos que se sueltan en el navegador."""
    job = store.create(name)
    copied = 0

    for upload in files:
        filename = Path(upload.filename or "").name
        if not filename or Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        destination = job.input_dir / filename
        with destination.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        copied += 1

    if copied == 0:
        store.delete(job.id)
        raise HTTPException(400, "ningun archivo con extension soportada")

    store.enqueue(job.id)
    return JSONResponse({"id": job.id, "archivos": copied})


@app.post("/api/jobs/folder")
async def create_from_folder(path: str = Form(...), name: str = Form("")) -> JSONResponse:
    """Crea un trabajo leyendo una carpeta local, sin copiar nada.

    Es la via recomendada para lotes grandes: un bracket de 90 RAW son varios
    gigas, y subirlos al propio ordenador para volver a escribirlos en disco es
    tiempo tirado.
    """
    source = Path(path).expanduser()
    if not source.is_dir():
        raise HTTPException(400, f"no existe la carpeta {source}")

    files = list_supported_files(source)
    if not files:
        raise HTTPException(400, f"no hay archivos procesables en {source}")

    job = store.create(name or source.name)
    # Enlaces duros cuando se puede: mismo disco, coste cero y sin duplicar
    # gigas de RAW. Si el origen esta en otro volumen, se copia.
    for file in files:
        destination = job.input_dir / file.name
        try:
            destination.hardlink_to(file)
        except (OSError, AttributeError):
            shutil.copy2(file, destination)

    store.enqueue(job.id)
    return JSONResponse({"id": job.id, "archivos": len(files)})


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "trabajo no encontrado")
    return JSONResponse(job.to_dict())


@app.get("/api/jobs/{job_id}/zip")
async def download_zip(job_id: str) -> FileResponse:
    job = store.get(job_id)
    if job is None or not job.zip_path.is_file():
        raise HTTPException(404, "todavia no hay resultado que descargar")
    return FileResponse(
        job.zip_path, media_type="application/zip", filename=f"{job.name}.zip"
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> JSONResponse:
    if not store.delete(job_id):
        raise HTTPException(404, "trabajo no encontrado")
    return JSONResponse({"ok": True})
