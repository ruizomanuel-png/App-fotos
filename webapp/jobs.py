"""Cola de trabajos de la app web.

Un unico worker en segundo plano procesa los lotes de uno en uno. No es una
limitacion a superar: revelar a resolucion completa satura la CPU y consume
varios gigas de RAM por escena, asi que dos lotes a la vez irian mas lento que
uno detras de otro, ademas de arriesgar quedarse sin memoria.

El estado se guarda en disco junto a cada trabajo, para que reiniciar la app no
haga perder lo ya procesado.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Queue

from hdrpipe.config import load_config
from hdrpipe.pipeline import run_batch

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "jobs"

STATE_PENDING = "en cola"
STATE_RUNNING = "procesando"
STATE_DONE = "terminado"
STATE_FAILED = "error"


@dataclass
class Job:
    id: str
    name: str
    created_at: str
    state: str = STATE_PENDING
    message: str = "esperando turno"
    done: int = 0
    total: int = 0
    delivered: int = 0
    review: int = 0
    errors: int = 0
    elapsed: float = 0.0
    detail: list[dict] = field(default_factory=list)

    @property
    def directory(self) -> Path:
        return DATA_DIR / self.id

    @property
    def input_dir(self) -> Path:
        return self.directory / "entrada"

    @property
    def output_dir(self) -> Path:
        return self.directory / "salida"

    @property
    def zip_path(self) -> Path:
        return self.directory / "resultado.zip"

    @property
    def progress(self) -> float:
        return round(100.0 * self.done / self.total, 1) if self.total else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["progress"] = self.progress
        data["zip_ready"] = self.zip_path.is_file()
        return data

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "estado.json").write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )


class JobStore:
    """Registro de trabajos con un worker en segundo plano."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._queue: Queue[str] = Queue()
        self._lock = threading.Lock()
        self._load_existing()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    # -- persistencia -----------------------------------------------------

    def _load_existing(self) -> None:
        for state_file in sorted(self.data_dir.glob("*/estado.json")):
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                job = Job(**data)
                # Un trabajo que estaba en marcha cuando se cerro la app no
                # puede continuar: se marca como fallido en lugar de quedarse
                # colgado en "procesando" para siempre.
                if job.state in {STATE_PENDING, STATE_RUNNING}:
                    job.state = STATE_FAILED
                    job.message = "interrumpido al cerrarse la aplicacion"
                self._jobs[job.id] = job
            except Exception as exc:  # noqa: BLE001 - un estado corrupto se ignora
                log.warning("no se pudo leer %s: %s", state_file, exc)

    # -- API --------------------------------------------------------------

    def create(self, name: str) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            name=name or "lote",
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        job.input_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[job.id] = job
        job.save()
        return job

    def enqueue(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            return
        job.state = STATE_PENDING
        job.message = "esperando turno"
        job.save()
        self._queue.put(job_id)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        shutil.rmtree(job.directory, ignore_errors=True)
        return True

    # -- worker -----------------------------------------------------------

    def _run_worker(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            try:
                self._process(job)
            except Exception as exc:  # noqa: BLE001 - el worker no puede morir
                log.exception("fallo el trabajo %s", job.id)
                job.state = STATE_FAILED
                job.message = f"{type(exc).__name__}: {exc}"
                job.save()

    def _process(self, job: Job) -> None:
        job.state = STATE_RUNNING
        job.message = "leyendo la carpeta"
        job.save()

        last_save = 0.0

        def progress(message: str, done: int, total: int) -> None:
            nonlocal last_save
            job.message, job.done, job.total = message, done, total
            # Escribir el estado en cada foto seria escribir en disco cientos de
            # veces; una vez por segundo basta para que la barra se mueva.
            now = time.monotonic()
            if now - last_save > 1.0:
                last_save = now
                job.save()

        cfg = load_config()
        result = run_batch(job.input_dir, job.output_dir, cfg, progress=progress)

        job.delivered = len(result.delivered)
        job.review = len(result.review)
        job.errors = len(result.errored)
        job.elapsed = round(result.elapsed_seconds, 1)
        job.detail = [
            {
                "nombre": scene.name,
                "entregada": scene.delivered,
                "motivos": scene.issues,
                "notas": scene.notes,
                "error": scene.error,
            }
            for scene in result.scenes
        ]

        if not result.scenes:
            job.state = STATE_FAILED
            job.message = "no se encontro ningun bracket en la carpeta"
        else:
            build_zip(job)
            job.state = STATE_DONE
            job.message = (
                f"{job.delivered} entregadas, {job.review} a revisar, {job.errors} con error"
            )
        job.done = job.total
        job.save()


def build_zip(job: Job) -> Path:
    """Empaqueta el resultado. `revisar/` va dentro pero en su propia carpeta."""
    with zipfile.ZipFile(job.zip_path, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(job.output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(job.output_dir))
    return job.zip_path


def purge_expired(store: JobStore, days: int) -> int:
    """Borra los trabajos caducados. Devuelve cuantos se han eliminado."""
    if days <= 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    removed = 0
    for job in store.list():
        try:
            created = datetime.fromisoformat(job.created_at)
        except ValueError:
            continue
        if created < cutoff and store.delete(job.id):
            removed += 1
            log.info("trabajo %s eliminado por antiguedad", job.id)
    return removed
