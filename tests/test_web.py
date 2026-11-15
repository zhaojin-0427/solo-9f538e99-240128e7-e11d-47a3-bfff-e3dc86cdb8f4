import io
import os
import re
import sqlite3
import threading
import time

import pytest

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import create_app, store


@pytest.fixture()
def app(tmp_path):
    application = create_app({"DATA_DIR": str(tmp_path / "data")})
    application.config.update(TESTING=True)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


def post_csv(client, study, payload, filename="visits.csv", follow=True):
    return client.post(
        "/upload",
        data={"study": study, "file": (io.BytesIO(payload), filename)},
        content_type="multipart/form-data",
        follow_redirects=follow,
    )


def flashes(response):
    body = response.data.decode()
    return [re.sub("<[^>]+>", "", text) for text in re.findall(r'<li class="flash.*?</li>', body, re.S)]


def test_index_and_static(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/healthz").get_json()["ok"] is True


def test_upload_download_roundtrip(client, app):
    payload = b"P1,2024-01-01T00:00:00Z,A00\nP2,2024-01-02T00:00:00Z,B00\n"
    response = post_csv(client, "研究甲", payload)
    assert response.status_code == 200
    assert "已生成替码文件" in "".join(flashes(response))

    db = store.connect(app.config["DATA_DIR"] + "/research.sqlite3")
    try:
        batch = store.fetch_batch(db, 1)
    finally:
        db.close()

    download = client.get("/results/" + batch.result_file)
    assert download.status_code == 200
    assert "attachment" in download.headers["Content-Disposition"]
    lines = download.data.decode().splitlines()
    assert [line.split(",", 1)[1] for line in lines] == [
        "2024-01-01T00:00:00Z,A00",
        "2024-01-02T00:00:00Z,B00",
    ]
    assert lines[0].split(",")[0].startswith("PSN-")


def test_pages_render(client):
    post_csv(client, "研究甲", b"P1,2024-01-01T00:00:00Z,A00\n")
    assert client.get("/batches").status_code == 200
    assert client.get("/batches?study_id=1").status_code == 200
    assert client.get("/studies/1").status_code == 200
    assert client.get("/batches/1").status_code == 200
    assert client.get("/batches/999").status_code == 404
    assert client.get("/studies/999").status_code == 404
    assert client.get("/results/nope.csv").status_code == 404


def test_rejected_batch_shows_message_and_leaves_nothing(client, app):
    response = post_csv(client, "研究甲", b"P1,2024-01-01T00:00:00Z,A00\nP2,2024-01-01T00:00:00Z,\n")
    assert "整批已拒绝" in "".join(flashes(response))
    db = sqlite3.connect(app.config["DATA_DIR"] + "/research.sqlite3")
    try:
        assert db.execute("select count(*) from batches").fetchone()[0] == 0
        assert db.execute("select count(*) from pseudonyms").fetchone()[0] == 0
    finally:
        db.close()
    assert os.listdir(app.config["DATA_DIR"] + "/results") == []
    assert os.listdir(app.config["DATA_DIR"] + "/tmp") == []


def test_blank_study_rejected(client):
    response = post_csv(client, "   ", b"P1,2024-01-01T00:00:00Z,A00\n")
    assert "研究项目不能为空" in "".join(flashes(response))


def test_non_csv_filename_rejected(client):
    response = post_csv(client, "S", b"P1,2024-01-01T00:00:00Z,A00\n", filename="visits.txt")
    assert "仅接受 .csv" in "".join(flashes(response))


def test_missing_file_rejected(client):
    response = client.post("/upload", data={"study": "S"}, follow_redirects=True)
    assert "请选择要上传的 CSV 文件" in "".join(flashes(response))


def test_download_requires_registration(app, client):
    data_dir = app.config["DATA_DIR"]
    orphan = os.path.join(data_dir, "results", "0" * 32 + ".csv")
    with open(orphan, "wb") as fh:
        fh.write(b"secret")
    assert client.get("/results/" + "0" * 32 + ".csv").status_code == 404


def test_restart_cleans_orphans_and_keeps_published(app, client):
    post_csv(client, "S", b"P1,2024-01-01T00:00:00Z,A00\n")
    data_dir = app.config["DATA_DIR"]
    published = os.listdir(os.path.join(data_dir, "results"))
    assert len(published) == 1
    orphan = os.path.join(data_dir, "results", "f" * 32 + ".csv")
    with open(orphan, "wb") as fh:
        fh.write(b"unregistered")

    restarted = create_app({"DATA_DIR": data_dir})
    restarted.config.update(TESTING=True)
    assert os.listdir(os.path.join(data_dir, "results")) == published
    resp = restarted.test_client().get("/results/" + published[0])
    assert resp.status_code == 200


def test_concurrent_same_patient_same_code(app):
    """Parallel uploads for one study must not create divergent codes."""
    payload = b"".join(
        ("P%d,2024-01-01T00:00:00Z,A00\n" % i).encode() for i in range(200)
    )
    results = []
    errors = []

    def worker(index):
        c = app.test_client()
        try:
            r = post_csv(c, "并发研究", payload, follow=False)
            results.append(r.headers.get("Location", ""))
        except Exception as exc:  # pragma: no cover - diagnostic
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    db = store.connect(app.config["DATA_DIR"] + "/research.sqlite3")
    try:
        count = db.execute("select count(*) from pseudonyms").fetchone()[0]
        batches = db.execute("select count(*) from batches").fetchone()[0]
    finally:
        db.close()
    assert batches == 8
    assert count == 200, "one patient must have exactly one code per study"


def test_two_studies_allocate_independently(client, app):
    post_csv(client, "甲", b"P1,2024-01-01T00:00:00Z,A00\n")
    post_csv(client, "乙", b"P1,2024-01-01T00:00:00Z,A00\n")
    db = store.connect(app.config["DATA_DIR"] + "/research.sqlite3")
    try:
        rows = db.execute(
            "select s.name, p.code from pseudonyms p join studies s using(study_id)"
            " order by s.name"
        ).fetchall()
    finally:
        db.close()
    assert len(rows) == 2
    assert rows[0][1] != rows[1][1]


def test_other_study_not_blocked_by_large_conversion(app):
    """A long conversion must not prevent another study's small upload."""
    data_dir = app.config["DATA_DIR"]
    slow_started = threading.Event()
    release_slow = threading.Event()

    class Trickle(io.RawIOBase):
        """Stand-in upload that stalls on demand, mimicking a huge file."""

        def __init__(self):
            self.parts = [b"P1,2024-01-01T00:00:00Z,A00\n"] * 50
            self.index = 0
            self.stalled = False

        def readable(self):
            return True

        def readline(self, limit=-1):
            if self.index >= 20 and not self.stalled:
                self.stalled = True
                slow_started.set()
                release_slow.wait(10)
            if self.index >= len(self.parts):
                return b""
            self.index += 1
            return self.parts[self.index - 1]

    from research.pipeline import convert_upload

    paths = store.Paths.build(data_dir)
    secret = store.load_secret(paths)
    errors = []

    def slow_worker():
        db = store.connect(data_dir + "/research.sqlite3")
        try:
            convert_upload(db, paths, secret, "大研究", Trickle(), max_bytes=10**9)
        except Exception as exc:  # pragma: no cover - diagnostic
            errors.append(exc)
        finally:
            db.close()

    thread = threading.Thread(target=slow_worker)
    thread.start()
    assert slow_started.wait(10)

    # While the large conversion is stalled, a different study completes.
    started = time.monotonic()
    response = post_csv(app.test_client(), "小研究", b"Q1,2024-02-01T00:00:00Z,B00\n")
    elapsed = time.monotonic() - started
    release_slow.set()
    thread.join(30)

    assert "已生成替码文件" in "".join(flashes(response))
    assert elapsed < 5, "small upload waited for the large one (%.2fs)" % elapsed
    assert not errors
