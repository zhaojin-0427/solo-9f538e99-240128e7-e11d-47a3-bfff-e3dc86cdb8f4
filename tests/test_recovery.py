"""Crash-recovery tests using real processes.

A batch is interrupted between "result file written" and "registry row
committed"; the next start-up must remove the unpublished file while the
previously published result stays downloadable.
"""

import io
import os
import sqlite3
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research import create_app, store  # noqa: E402

CRASH_SCRIPT = textwrap.dedent(
    """
    import io, os, sys
    sys.path.insert(0, {root!r})
    from research import store
    from research.pipeline import convert_upload

    data_dir, study, payload = sys.argv[1], sys.argv[2], sys.argv[3].encode()

    # Die inside the publication transaction, after the file was published.
    def die(*a, **k):
        os._exit(9)

    store.commit_batch = die
    paths = store.Paths.build(data_dir)
    store.initialize(paths)
    secret = store.load_secret(paths)
    conn = store.connect(paths.db_path)
    convert_upload(conn, paths, secret, study, io.BytesIO(payload), max_bytes=1 << 26)
    """
)


def run_crash(data_dir, study, payload):
    proc = subprocess.run(
        [sys.executable, "-c", CRASH_SCRIPT.format(root=ROOT), data_dir, study, payload],
        capture_output=True,
    )
    assert proc.returncode == -9 or proc.returncode == 9, proc.stderr


def test_crash_between_write_and_register(tmp_path):
    data_dir = str(tmp_path / "data")
    app = create_app({"DATA_DIR": data_dir})
    app.config.update(TESTING=True)
    client = app.test_client()

    good = b"P1,2024-01-01T00:00:00Z,A00\nP2,2024-01-02T00:00:00Z,B00\n"
    client.post(
        "/upload",
        data={"study": "研究甲", "file": (io.BytesIO(good), "a.csv")},
        content_type="multipart/form-data",
    )
    db = sqlite3.connect(os.path.join(data_dir, "research.sqlite3"))
    old_file = db.execute("SELECT result_file FROM batches").fetchone()[0]
    db.close()
    results_dir = os.path.join(data_dir, "results")
    assert os.listdir(results_dir) == [old_file]

    # Simulate a crash while publishing a second batch.
    run_crash(data_dir, "研究甲", good)
    crashed = os.listdir(results_dir)
    assert len(crashed) == 2, "the crash must leave the half-published file"
    assert os.listdir(os.path.join(data_dir, "tmp")), "work dir should survive the crash"

    # Restart: the unpublished file and stale work dir are swept away.
    restarted = create_app({"DATA_DIR": data_dir})
    restarted.config.update(TESTING=True)
    assert os.listdir(results_dir) == [old_file]
    assert os.listdir(os.path.join(data_dir, "tmp")) == []

    fresh = restarted.test_client()
    download = fresh.get("/results/" + old_file)
    assert download.status_code == 200
    text = download.data.decode()
    assert text.count("\n") == 2
    assert "PSN-" in text

    # No ghost batch rows were created by the crashed run.
    db = sqlite3.connect(os.path.join(data_dir, "research.sqlite3"))
    assert db.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 1
    codes = db.execute("SELECT COUNT(*) FROM pseudonyms").fetchone()[0]
    db.close()
    assert codes == 2


def test_crash_does_not_lose_registered_codes(tmp_path):
    """Codes staged by a crashed batch must not leak into the registry."""
    data_dir = str(tmp_path / "data")
    paths = store.Paths.build(data_dir)
    store.initialize(paths)
    secret = store.load_secret(paths)
    conn = store.connect(paths.db_path)

    # A normal batch registers two patients.
    from research.pipeline import convert_upload

    convert_upload(
        conn, paths, secret, "S", io.BytesIO(b"P1,2024-01-01T00:00:00Z,A\nP2,2024-01-01T00:00:00Z,B\n"),
        max_bytes=1 << 20,
    )
    first = dict(
        conn.execute(
            "SELECT p.patient_id, p.code FROM pseudonyms p JOIN studies s USING(study_id)"
            " WHERE s.name='S'"
        ).fetchall()
    )

    # The crashed run stages P3 but never commits.
    run_crash(data_dir, "S", b"P1,2024-01-01T00:00:00Z,A\nP3,2024-01-01T00:00:00Z,C\n")
    create_app({"DATA_DIR": data_dir})

    conn2 = store.connect(paths.db_path)
    after = dict(
        conn2.execute(
            "SELECT p.patient_id, p.code FROM pseudonyms p JOIN studies s USING(study_id)"
            " WHERE s.name='S'"
        ).fetchall()
    )
    assert after == first
    conn.close()
    conn2.close()
