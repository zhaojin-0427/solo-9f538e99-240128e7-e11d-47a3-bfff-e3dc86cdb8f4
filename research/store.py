"""Durable SQLite registry, pseudonym derivation and result publication.

Publication protocol
--------------------
1. The upload is streamed through validation straight into
   ``<data>/tmp/work-<rand>/masked.csv``.  New pseudonyms are collected in a
   throw-away SQLite file inside the same work directory, so a rejected batch
   costs at most one work directory that is deleted on the way out.
2. The masked CSV is fsync-ed and atomically renamed to
   ``<data>/results/<32 hex>.csv``.  The random name is not yet known to any
   client, and downloads are only served for registered batches, so the file
   is not reachable yet.
3. A single ``BEGIN IMMEDIATE`` transaction creates the study row if needed,
   copies staged pseudonyms into ``pseudonyms`` and inserts the ``batches``
   row.  This commit is the publication point: new codes and the downloadable
   result become visible together.
4. On any failure the transaction rolls back and the result file is removed.

A crash between step 2 and step 3 leaves an unreferenced file; at the next
start :func:`cleanup_unpublished` deletes it and any stale work directory.
Registered results and committed codes are never touched.

Because codes are cryptographically derived from a per-installation secret
plus the study name and the patient id, two concurrent requests can compute
only one code per (study, patient); the ``PRIMARY KEY (study_id, patient_id)``
constraint and the verification query inside the publication transaction are
the final authority.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import re
import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass

RESULT_NAME_RE = re.compile(r"^[0-9a-f]{32}\.csv$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
    study_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

-- Code uniqueness per study follows from keyed derivation (80-bit
-- digests); the primary key is what enforces "one code per patient".
CREATE TABLE IF NOT EXISTS pseudonyms (
    study_id   INTEGER NOT NULL REFERENCES studies(study_id),
    patient_id TEXT NOT NULL,
    code       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (study_id, patient_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS batches (
    batch_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    study_id    INTEGER NOT NULL REFERENCES studies(study_id),
    result_file TEXT NOT NULL UNIQUE,
    rows        INTEGER NOT NULL,
    bytes_in    INTEGER NOT NULL,
    bytes_out   INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_batches_study ON batches(study_id, batch_id DESC);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

STAGING_SCHEMA = """
CREATE TABLE new_codes (
    patient_id TEXT PRIMARY KEY,
    code       TEXT NOT NULL
);
"""


class ConflictError(Exception):
    """A stored pseudonym disagrees with the derived one."""


class StorageError(Exception):
    """Persistence failed; the batch must not be published."""


# ---------------------------------------------------------------------------
# Paths and secret
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    data_dir: str
    db_path: str
    results_dir: str
    tmp_dir: str
    secret_path: str

    @classmethod
    def build(cls, data_dir: str) -> "Paths":
        data_dir = os.path.abspath(data_dir)
        return cls(
            data_dir=data_dir,
            db_path=os.path.join(data_dir, "research.sqlite3"),
            results_dir=os.path.join(data_dir, "results"),
            tmp_dir=os.path.join(data_dir, "tmp"),
            secret_path=os.path.join(data_dir, "secret.key"),
        )

    def ensure(self) -> None:
        for path in (self.data_dir, self.results_dir, self.tmp_dir):
            os.makedirs(path, mode=0o750, exist_ok=True)


def load_secret(paths: Paths) -> bytes:
    """Return the installation secret, creating it once if necessary.

    ``PSEUDONYM_SECRET`` may be supplied by the deployment (must be stable
    across restarts, otherwise already-issued codes cannot be reproduced).
    """
    env = os.environ.get("PSEUDONYM_SECRET")
    if env:
        secret = env.encode("utf-8")
        if len(secret) < 16:
            raise RuntimeError("PSEUDONYM_SECRET 至少需要 16 字节")
        return secret
    try:
        with open(paths.secret_path, "rb") as fh:
            secret = fh.read()
        if len(secret) < 32:
            raise RuntimeError("secret.key 内容异常")
        return secret
    except FileNotFoundError:
        pass
    secret = os.urandom(32)
    try:
        fd = os.open(paths.secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        with open(paths.secret_path, "rb") as fh:
            return fh.read()
    with os.fdopen(fd, "wb") as fh:
        fh.write(secret)
        fh.flush()
        os.fsync(fh.fileno())
    return secret


def load_flash_secret(paths: Paths) -> bytes:
    """Key for the session cookie holding flash messages.

    Deliberately derived from (not equal to) the pseudonym secret so a leaked
    session key cannot be used to reproduce pseudonyms.
    """
    env = os.environ.get("SESSION_SECRET")
    if env:
        return env.encode("utf-8")
    return hashlib.sha256(b"session-key:" + load_secret(paths)).digest()


_PERSON = b"psn-v1"


def study_key(secret: bytes, study_name: str) -> bytes:
    """Per-study key, so different studies never share a code space."""
    return hashlib.blake2b(
        study_name.encode("utf-8"), key=secret, digest_size=32, person=_PERSON
    ).digest()


def code_hasher(secret: bytes, study_name: str):
    """Return ``f(patient_id) -> code`` for one study.

    A prepared keyed BLAKE2b state is copied per patient (cheap) instead of
    constructing a fresh HMAC for every row.
    """
    base = hashlib.blake2b(
        digest_size=10, key=study_key(secret, study_name), person=_PERSON
    )

    def derive(patient_id: str) -> str:
        h = base.copy()
        h.update(patient_id.encode("utf-8"))
        return "PSN-" + h.hexdigest().upper()

    return derive


def derive_code(secret: bytes, study_name: str, patient_id: str) -> str:
    """Deterministic, keyed pseudonym for one (study, patient) pair.

    Equivalent to ``code_hasher(secret, study_name)(patient_id)``.
    """
    h = hashlib.blake2b(
        digest_size=10, key=study_key(secret, study_name), person=_PERSON
    )
    h.update(patient_id.encode("utf-8"))
    return "PSN-" + h.hexdigest().upper()


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=120000")
    return conn


def initialize(paths: Paths) -> None:
    paths.ensure()
    conn = connect(paths.db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('schema', '1')")
    finally:
        conn.close()


def database_ready(conn: sqlite3.Connection) -> bool:
    """Cheap probe used by the health endpoint and writes."""
    try:
        return conn.execute("SELECT 1 FROM meta WHERE key='schema'").fetchone() is not None
    except sqlite3.Error:
        return False


def cleanup_unpublished(paths: Paths, conn: sqlite3.Connection) -> dict:
    """Delete result files with no batch row and stale work directories.

    Runs at start-up, before requests are served, so an interrupted
    file-write/registration never leaves a downloadable orphan.  Registered
    results are never touched, so earlier downloads stay available.
    """
    with publish_lock(paths, exclusive=True):
        return _cleanup_unpublished_locked(paths, conn)


def _cleanup_unpublished_locked(paths: Paths, conn: sqlite3.Connection) -> dict:
    removed_files: list[str] = []
    known = {row[0] for row in conn.execute("SELECT result_file FROM batches")}
    for name in sorted(os.listdir(paths.results_dir)):
        full = os.path.join(paths.results_dir, name)
        if not os.path.isfile(full):
            continue
        if name in known:
            continue
        try:
            os.unlink(full)
            removed_files.append(name)
        except OSError:
            pass
    removed_dirs = 0
    for name in os.listdir(paths.tmp_dir):
        full = os.path.join(paths.tmp_dir, name)
        try:
            if os.path.isdir(full) and not os.path.islink(full):
                shutil.rmtree(full, ignore_errors=True)
            else:
                os.unlink(full)
            removed_dirs += 1
        except OSError:
            pass
    return {"result_files": removed_files, "tmp_entries": removed_dirs}


# ---------------------------------------------------------------------------
# Cross-process publication lock
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def publish_lock(paths: Paths, exclusive: bool = False):
    """Coordinate result-file publication with start-up cleanup.

    Publishers take a shared lock around "write file, then register it", so a
    cleanup running in another process cannot delete a file that is about to
    be registered.  Cleanup takes the exclusive lock.
    """
    lock_path = os.path.join(paths.data_dir, "publish.lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o640)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# Staging database
# ---------------------------------------------------------------------------

class Staging:
    """Disposable SQLite file holding this batch's newly derived codes."""

    def __init__(self, paths: Paths):
        paths.ensure()
        self.dir = os.path.join(paths.tmp_dir, "work-" + uuid.uuid4().hex)
        os.makedirs(self.dir, mode=0o700)
        self.path = os.path.join(self.dir, "codes.sqlite3")
        self.work_csv = os.path.join(self.dir, "masked.csv")
        self.conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        # Disposable data: losing it to a crash is exactly what we want.
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA cache_size=-2048")
        self.conn.executescript(STAGING_SCHEMA)
        self.conn.execute("BEGIN")
        self._uncommitted = 0
        self._buffer: list[tuple[str, str]] = []
        self._sealed = False
        self.count = 0

    def add(self, patient_id: str, code: str) -> None:
        """Queue a derived code for the batch.

        Writes are grouped so a multi-million-row file does not pay one
        SQLite round-trip per row.  Duplicates are ignored: the code is
        derived deterministically, so a repeated patient always yields the
        same value.
        """
        self._buffer.append((patient_id, code))
        self.count += 1
        if len(self._buffer) >= 5000:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        self.conn.executemany(
            "INSERT OR IGNORE INTO new_codes(patient_id, code) VALUES(?, ?)",
            self._buffer,
        )
        self._buffer.clear()
        self._uncommitted += 1
        if self._uncommitted >= 20:
            self.conn.execute("COMMIT")
            self.conn.execute("BEGIN")
            self._uncommitted = 0

    def seal(self) -> None:
        """Flush buffered codes and close the staging database (idempotent)."""
        if self._sealed:
            return
        self._sealed = True
        self._flush()
        if self.conn.in_transaction:
            self.conn.execute("COMMIT")
        self.conn.close()

    def discard(self) -> None:
        if not self._sealed:
            self._sealed = True
            try:
                self.conn.close()
            except sqlite3.Error:
                pass
        shutil.rmtree(self.dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PublishedBatch:
    batch_id: int
    study_id: int
    study_name: str
    result_file: str
    rows: int
    bytes_in: int
    bytes_out: int
    sha256: str
    created_at: str


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save_result_file(paths: Paths, work_csv: str) -> str:
    """Atomically move the finished CSV into the results directory."""
    name = uuid.uuid4().hex + ".csv"
    os.replace(work_csv, os.path.join(paths.results_dir, name))
    _fsync_dir(paths.results_dir)
    return name


def is_registered(conn: sqlite3.Connection, name: str) -> bool:
    """True when a batch row already references this result file."""
    try:
        return conn.execute(
            "SELECT 1 FROM batches WHERE result_file = ?", (name,)
        ).fetchone() is not None
    except sqlite3.Error:
        return False


def remove_result_file(paths: Paths, name: str) -> None:
    if not RESULT_NAME_RE.match(name):
        return
    try:
        os.unlink(os.path.join(paths.results_dir, name))
    except OSError:
        pass


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def commit_batch(
    conn: sqlite3.Connection,
    staging: Staging,
    result_file: str,
    study_name: str,
    rows: int,
    bytes_in: int,
    bytes_out: int,
    sha256: str,
) -> tuple[int, int]:
    """Register a batch and its new pseudonyms in one transaction.

    Returns ``(batch_id, inserted_codes)``; ``inserted_codes`` is the number
    of genuinely new ``(study, patient)`` rows written, which is what the UI
    reports as "新增替码".
    """
    if not RESULT_NAME_RE.match(result_file):
        raise StorageError("非法结果文件名")
    staging.seal()
    alias = "stage_" + uuid.uuid4().hex[:12]
    attached = False
    try:
        conn.execute('ATTACH DATABASE ? AS "%s"' % alias, (staging.path,))
        attached = True
        conn.execute("BEGIN IMMEDIATE")
        now = utc_now()
        conn.execute(
            "INSERT OR IGNORE INTO studies(name, created_at) VALUES(?, ?)",
            (study_name, now),
        )
        study_id = conn.execute(
            "SELECT study_id FROM studies WHERE name = ?", (study_name,)
        ).fetchone()[0]
        insert_sql = (
            "INSERT OR IGNORE INTO pseudonyms(study_id, patient_id, code, created_at)"
            ' SELECT ?, patient_id, code, ? FROM "%s".new_codes' % alias
        )
        inserted = conn.execute(insert_sql, (study_id, now)).rowcount
        verify_sql = (
            'SELECT COUNT(*) FROM "%s".new_codes nc'
            " LEFT JOIN pseudonyms p"
            "   ON p.study_id = ? AND p.patient_id = nc.patient_id"
            " WHERE p.code IS NULL OR p.code <> nc.code" % alias
        )
        mismatched = conn.execute(verify_sql, (study_id,)).fetchone()[0]
        if mismatched:
            raise ConflictError("同一研究同一患者已存在不同替码")
        batch_id = conn.execute(
            "INSERT INTO batches(study_id, result_file, rows, bytes_in, bytes_out,"
            " sha256, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (study_id, result_file, rows, bytes_in, bytes_out, sha256, now),
        ).lastrowid
        conn.execute("COMMIT")
        return batch_id, max(inserted, 0)
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        if attached:
            try:
                conn.execute('DETACH DATABASE "%s"' % alias)
            except sqlite3.Error:
                # Leave the connection for the pool to discard; a later
                # request opens a fresh one.
                pass


def fetch_batch(conn: sqlite3.Connection, batch_id: int) -> PublishedBatch | None:
    row = conn.execute(
        "SELECT b.batch_id, b.study_id, s.name, b.result_file, b.rows, b.bytes_in,"
        " b.bytes_out, b.sha256, b.created_at FROM batches b"
        " JOIN studies s ON s.study_id = b.study_id WHERE b.batch_id = ?",
        (batch_id,),
    ).fetchone()
    return PublishedBatch(*row) if row else None


def list_studies(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT s.study_id, s.name, s.created_at,"
        " (SELECT COUNT(*) FROM pseudonyms p WHERE p.study_id = s.study_id),"
        " (SELECT COUNT(*) FROM batches b WHERE b.study_id = s.study_id)"
        " FROM studies s ORDER BY s.study_id"
    ).fetchall()
    return [
        {
            "study_id": r[0],
            "name": r[1],
            "created_at": r[2],
            "patients": r[3],
            "batches": r[4],
        }
        for r in rows
    ]


def fetch_study(conn: sqlite3.Connection, study_id: int) -> dict | None:
    row = conn.execute(
        "SELECT study_id, name, created_at FROM studies WHERE study_id = ?", (study_id,)
    ).fetchone()
    if not row:
        return None
    patients = conn.execute(
        "SELECT COUNT(*) FROM pseudonyms WHERE study_id = ?", (study_id,)
    ).fetchone()[0]
    batches = conn.execute(
        "SELECT COUNT(*) FROM batches WHERE study_id = ?", (study_id,)
    ).fetchone()[0]
    return {
        "study_id": row[0],
        "name": row[1],
        "created_at": row[2],
        "patients": patients,
        "batches": batches,
    }


def list_batches(conn: sqlite3.Connection, study_id: int | None = None, limit: int = 50) -> list[dict]:
    sql = (
        "SELECT b.batch_id, b.study_id, s.name, b.rows, b.bytes_out, b.created_at,"
        " b.result_file FROM batches b JOIN studies s ON s.study_id = b.study_id"
    )
    params: list = []
    if study_id is not None:
        sql += " WHERE b.study_id = ?"
        params.append(study_id)
    sql += " ORDER BY b.batch_id DESC LIMIT ?"
    params.append(limit)
    return [
        {
            "batch_id": r[0],
            "study_id": r[1],
            "study": r[2],
            "rows": r[3],
            "bytes_out": r[4],
            "created_at": r[5],
            "result_file": r[6],
        }
        for r in conn.execute(sql, params).fetchall()
    ]
