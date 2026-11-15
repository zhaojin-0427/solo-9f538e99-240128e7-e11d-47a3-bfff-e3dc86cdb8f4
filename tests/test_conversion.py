import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import store
from research.pipeline import convert_upload
from research.validation import (
    FileTooLarge,
    InvalidBatch,
    validate_visit_utc,
    normalize_study_name,
)


@pytest.fixture()
def env(tmp_path):
    paths = store.Paths.build(str(tmp_path / "data"))
    store.initialize(paths)
    secret = store.load_secret(paths)
    conn = store.connect(paths.db_path)
    yield paths, secret, conn
    conn.close()


def upload(env, study, payload, name="in.csv"):
    paths, secret, conn = env
    return convert_upload(
        conn, paths, secret, study, io.BytesIO(payload), max_bytes=256 * 1024 * 1024
    )


def read_result(env, result):
    paths, _, _ = env
    with open(os.path.join(paths.results_dir, result.result_file), "rb") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    [
        "2024-05-01T09:30:00Z",
        "2024-05-01T09:30Z",
        "2024-02-29T23:59:59Z",
        "2024-05-01 09:30:00+00:00",
        "2024-05-01T09:30:00.123456UTC",
        "1999-12-31T00:00:00z",
    ],
)
def test_valid_timestamps(value):
    assert validate_visit_utc(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "2023-02-29T00:00:00Z",     # not a leap year
        "2024-13-01T00:00:00Z",     # month
        "2024-04-31T00:00:00Z",     # day
        "2024-05-01T24:00:00Z",     # hour
        "2024-05-01T09:60:00Z",     # minute
        "2024-05-01T09:30:60Z",     # second
        "2024-05-01",               # date only
        "2024-05-01T09:30:00+08:00",  # not UTC
        "01/05/2024 09:30",         # wrong format
        "not-a-time",
        "0000-01-01T00:00:00Z",     # year 0
    ],
)
def test_invalid_timestamps(value):
    with pytest.raises(ValueError):
        validate_visit_utc(value)


def test_study_name_normalisation():
    assert normalize_study_name("  糖尿病   随访 ") == "糖尿病 随访"
    with pytest.raises(Exception):
        normalize_study_name("   ")


# ---------------------------------------------------------------------------
# Happy path and pseudonym stability
# ---------------------------------------------------------------------------

def test_one_code_per_patient_across_batches_and_studies(env):
    first = upload(env, "Study A", b"P1,2024-01-01T00:00:00Z,A00\nP2,2024-01-02T00:00:00Z,B00\n")
    second = upload(env, "Study A", b"P1,2024-03-01T00:00:00Z,C00\n")
    other = upload(env, "Study B", b"P1,2024-01-01T00:00:00Z,A00\n")

    lines1 = read_result(env, first).decode().splitlines()
    lines2 = read_result(env, second).decode().splitlines()
    lines3 = read_result(env, other).decode().splitlines()
    code_p1 = lines1[0].split(",")[0]
    assert lines2[0].split(",")[0] == code_p1
    assert lines3[0].split(",")[0] != code_p1
    assert lines1[0].split(",")[1:] == ["2024-01-01T00:00:00Z", "A00"]
    assert first.stats.new_codes == 2 and second.stats.new_codes == 0
    assert first.stats.rows == 2 and second.stats.reused_rows == 1


def test_patient_within_batch_gets_single_code(env):
    result = upload(env, "S", b"P1,2024-01-01T00:00:00Z,A00\nP1,2024-01-02T00:00:00Z,A01\n")
    codes = {line.split(",")[0] for line in read_result(env, result).decode().splitlines()}
    assert len(codes) == 1
    assert result.stats.new_codes == 1


def test_row_order_and_content_preserved(env):
    payload = (
        b"alpha,2024-05-01T09:30:00Z,J18.9\n"
        b"beta,2024-05-02T10:00:00Z,E11.9\n"
        b"gamma,2024-05-03T11:15:00Z,I10\n"
    )
    result = upload(env, "S", payload)
    out = read_result(env, result).decode().splitlines()
    assert len(out) == 3
    expected = [chunk.decode().split(",", 1)[1] for chunk in payload.splitlines()]
    assert [line.split(",", 1)[1] for line in out] == expected
    assert len({line.split(",")[0] for line in out}) == 3


def test_crlf_and_bom_preserved(env):
    payload = b"\xef\xbb\xbfP1,2024-01-01T00:00:00Z,A00\r\nP2,2024-01-02T00:00:00Z,B00\r\n"
    result = upload(env, "S", payload)
    raw = read_result(env, result)
    assert raw.startswith(b"\xef\xbb\xbf")
    assert raw.count(b"\r\n") == 2


def test_header_row_passed_through(env):
    payload = b"patient_id,visit_utc,diagnosis_code\nP1,2024-01-01T00:00:00Z,A00\n"
    result = upload(env, "S", payload)
    out = read_result(env, result).decode().splitlines()
    assert out[0] == "patient_id,visit_utc,diagnosis_code"
    assert out[1].endswith(",2024-01-01T00:00:00Z,A00")


def test_quoted_fields(env):
    payload = b'"P,1","2024-01-01T00:00:00Z","A""01"\n'
    result = upload(env, "S", payload)
    out = read_result(env, result).decode()
    assert out.endswith(',"2024-01-01T00:00:00Z","A""01"\n')


def test_unicode_fields(env):
    payload = "患者甲,2024-01-01T00:00:00Z,糖尿病\n".encode()
    result = upload(env, "S", payload)
    assert read_result(env, result).decode().split(",")[1:] == ["2024-01-01T00:00:00Z", "糖尿病\n"]


def test_field_at_limit_accepted(env):
    diag = "D" * 100
    result = upload(env, "S", ("P1,2024-01-01T00:00:00Z,%s\n" % diag).encode())
    assert result.stats.rows == 1


# ---------------------------------------------------------------------------
# Rejection leaves nothing behind
# ---------------------------------------------------------------------------

def _snapshot(env):
    paths, _, conn = env
    return (
        sorted(os.listdir(paths.results_dir)),
        sorted(os.listdir(paths.tmp_dir)),
        conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM pseudonyms").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM studies").fetchone()[0],
    )


@pytest.mark.parametrize(
    "payload, reason",
    [
        (b"P1,2024-01-01T00:00:00Z,A00\n,2024-01-02T00:00:00Z,B00\n", "患者号缺失"),
        (b"P1,2024-01-01T00:00:00Z,A00\nP2,,B00\n", "就诊 UTC 时间缺失"),
        (b" P1,2024-01-01T00:00:00Z,A00\n", "首尾存在空白字符"),
        (b"P1,2024-01-01T00:00:00Z,A00\nP2,2024-01-02T00:00:00Z,\n", "诊断码缺失"),
        (b"P1,2024-01-01T00:00:00Z,A00\nP2,2024-01-02T00:00:00Z,A,B\n", "3 列"),
        (b"P1,2024-01-01T00:00:00Z,A00\nP2,2024-01-02T00:00:00Z,A00\nP3,2024-01-03T00:00:00Z,A00,", "3 列"),
        (b"P1,2024-01-01T00:00:00Z\n", "3 列"),
        (b"P1,2024-02-30T00:00:00Z,A00\n", "时间非法"),
        (b"P1,2024-01-01T00:00:00,A00\n", "时间非法"),
        (b"\n", "空行"),
        (b"P1,2024-01-01T00:00:00Z,A00\n\nP2,2024-01-02T00:00:00Z,B00\n", "空行"),
        (b"", "没有可转换的数据行"),
        (b"patient_id,visit_utc,diagnosis_code\n", "没有可转换的数据行"),
    ],
)
def test_invalid_batches_leave_no_trace(env, payload, reason):
    before = _snapshot(env)
    with pytest.raises(InvalidBatch) as info:
        upload(env, "S", payload)
    assert reason in info.value.message
    assert _snapshot(env) == before


def test_invalid_batch_on_new_study_creates_no_study(env):
    before = _snapshot(env)
    with pytest.raises(InvalidBatch):
        upload(env, "Brand New", b"P1,2024-01-01T00:00:00Z,A00\nP2,2024-01-01T00:00:00Z,\n")
    assert _snapshot(env) == before


def test_invalid_row_mid_file_rejects_everything(env):
    payload = b"".join(
        ("P%d,2024-01-01T00:00:00Z,A00\n" % i).encode() for i in range(50)
    ) + b"P50,2024-13-01T00:00:00Z,A00\nP99,2024-01-02T00:00:00Z,B00\n"
    before = _snapshot(env)
    with pytest.raises(InvalidBatch):
        upload(env, "S", payload)
    assert _snapshot(env) == before


def test_overlong_field_rejected(env):
    payload = ("P1,2024-01-01T00:00:00Z,%s\n" % ("D" * 101)).encode()
    before = _snapshot(env)
    with pytest.raises(InvalidBatch) as info:
        upload(env, "S", payload)
    assert "100" in info.value.message
    assert _snapshot(env) == before


def test_non_utf8_rejected(env):
    before = _snapshot(env)
    with pytest.raises(InvalidBatch):
        upload(env, "S", b"P1,2024-01-01T00:00:00Z,A00\n\xff\xfe,2024-01-02T00:00:00Z,B00\n")
    assert _snapshot(env) == before


def test_oversized_upload_rejected(env):
    paths, secret, conn = env
    payload = b"P1,2024-01-01T00:00:00Z,A00\n" * 100
    with pytest.raises(FileTooLarge):
        convert_upload(conn, paths, secret, "S", io.BytesIO(payload), max_bytes=100)
    assert sorted(os.listdir(paths.results_dir)) == []
    assert conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Failure injection: file written but registration fails
# ---------------------------------------------------------------------------

def test_registration_failure_removes_result_and_codes(env, monkeypatch):
    paths, secret, conn = env

    def boom(*args, **kwargs):
        raise __import__("research.store", fromlist=["StorageError"]).StorageError("disk full")

    monkeypatch.setattr(store, "commit_batch", boom)
    before = _snapshot(env)
    with pytest.raises(Exception):
        upload(env, "S", b"P1,2024-01-01T00:00:00Z,A00\n")
    assert _snapshot(env) == before


def test_crash_between_file_and_registration_is_cleaned(env):
    paths, _, conn = env
    # Simulate the crash: file published to results, no row inserted.
    orphan = os.path.join(paths.results_dir, "deadbeef" * 4 + ".csv")
    with open(orphan, "wb") as fh:
        fh.write(b"leftover")
    stale = os.path.join(paths.tmp_dir, "work-stale")
    os.makedirs(stale)
    with open(os.path.join(stale, "masked.csv"), "wb") as fh:
        fh.write(b"partial")

    result = upload(env, "S", b"P1,2024-01-01T00:00:00Z,A00\n")
    before = {result.result_file}
    swept = store.cleanup_unpublished(paths, conn)
    assert swept["result_files"] == [orphan.rsplit("/", 1)[-1]]
    assert not os.path.exists(orphan)
    assert not os.path.exists(stale)
    # The registered result survives, so earlier downloads remain available.
    assert os.path.exists(os.path.join(paths.results_dir, result.result_file))
    assert sorted(os.listdir(paths.results_dir)) == sorted(before)


def test_secret_is_stable_across_restart(env):
    paths, secret, _ = env
    assert store.load_secret(paths) == secret
    assert store.load_secret(store.Paths.build(paths.data_dir)) == secret
