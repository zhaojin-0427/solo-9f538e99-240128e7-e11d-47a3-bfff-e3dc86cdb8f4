import io
import os
import sqlite3
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import create_app, store
from research.analysis import (
    AnalysisError,
    Member,
    build_cohorts,
    match_cohorts,
    parse_targets,
)
from research.validation import parse_visit_utc


def M(code, day):
    return Member(code, date.fromisoformat(day))


# ---------------------------------------------------------------------------
# 最优配对
# ---------------------------------------------------------------------------

def test_match_unique_optimum():
    cases = [M("C1", "2024-05-01"), M("C2", "2024-05-03")]
    controls = [M("K1", "2024-05-02"), M("K2", "2024-05-10"), M("K3", "2024-06-03")]
    pairs, uc, uk, total, possible, maybe_unpaired = match_cohorts(cases, controls)
    # C1-K1(1) + C2-K2(7) 是唯一最优；K3 超出窗口。
    assert pairs == [(0, 0), (1, 1)]
    assert total == 8
    assert uc == [] and uk == [2]
    assert possible == [[0], [1]]
    assert maybe_unpaired == [False, False]


def test_match_window_boundary():
    cases = [M("C1", "2024-05-08")]
    controls = [M("K1", "2024-05-01"), M("K2", "2024-05-16")]
    # K1 恰好 7 天可配对；K2 相差 8 天不可。
    pairs, uc, uk, total, possible, maybe_unpaired = match_cohorts(cases, controls)
    assert pairs == [(0, 0)]
    assert total == 7
    assert uk == [1]
    assert possible == [[0]]


def test_match_prefers_more_pairs_over_smaller_weight():
    cases = [M("C1", "2024-05-01"), M("C2", "2024-05-01")]
    controls = [M("K1", "2024-05-01"), M("K2", "2024-05-08")]
    # 两对（总日差 7）优于一对（总日差 0）。
    pairs, uc, uk, total, possible, maybe_unpaired = match_cohorts(cases, controls)
    assert len(pairs) == 2
    assert total == 7
    assert possible == [[0, 1], [0, 1]]
    assert maybe_unpaired == [False, False]


def test_match_min_weight_among_max_cardinality():
    cases = [M("C1", "2024-05-01"), M("C2", "2024-05-02")]
    controls = [M("K1", "2024-05-01"), M("K2", "2024-05-02")]
    # 两种最大配对：日差 0+0 与 1+1，必须选前者。
    pairs, uc, uk, total, possible, maybe_unpaired = match_cohorts(cases, controls)
    assert pairs == [(0, 0), (1, 1)]
    assert total == 0
    assert possible == [[0], [1]]
    assert maybe_unpaired == [False, False]


def test_match_all_ties_means_uncertain():
    cases = [M("C1", "2024-05-01"), M("C2", "2024-05-01")]
    controls = [M("K1", "2024-05-01"), M("K2", "2024-05-01")]
    pairs, uc, uk, total, possible, maybe_unpaired = match_cohorts(cases, controls)
    assert len(pairs) == 2 and total == 0
    assert possible == [[0, 1], [0, 1]]
    assert maybe_unpaired == [False, False]


def test_match_maybe_unpaired_when_controls_scarce():
    cases = [M("C1", "2024-05-01"), M("C2", "2024-05-01")]
    controls = [M("K1", "2024-05-01")]
    pairs, uc, uk, total, possible, maybe_unpaired = match_cohorts(cases, controls)
    assert len(pairs) == 1
    assert possible == [[0], [0]]
    assert maybe_unpaired == [True, True]


def test_match_no_edges():
    cases = [M("C1", "2024-05-01")]
    controls = [M("K1", "2024-06-01")]
    pairs, uc, uk, total, possible, maybe_unpaired = match_cohorts(cases, controls)
    assert pairs == [] and total == 0
    assert uc == [0] and uk == [0]
    assert possible == [[]]
    assert maybe_unpaired == [True]


def test_match_empty_sides():
    pairs, uc, uk, total, possible, maybe_unpaired = match_cohorts(
        [M("C1", "2024-05-01")], []
    )
    assert pairs == [] and uc == [0] and uk == []
    assert possible == [[]] and maybe_unpaired == [True]
    pairs, uc, uk, total, possible, maybe_unpaired = match_cohorts(
        [], [M("K1", "2024-05-01")]
    )
    assert pairs == [] and uc == [] and uk == [0]
    assert possible == [] and maybe_unpaired == []


def test_match_bruteforce_crosscheck():
    """与暴力枚举所有配对对照：最优值与“可能的对照/可能未配对”一致。"""
    import itertools

    case_days = ["2024-05-01", "2024-05-04", "2024-05-06", "2024-05-09"]
    control_days = ["2024-05-02", "2024-05-03", "2024-05-08", "2024-05-10", "2024-05-12"]
    cases = [M("C%d" % i, d) for i, d in enumerate(case_days)]
    controls = [M("K%d" % j, d) for j, d in enumerate(control_days)]
    pairs, uc, uk, total, possible, maybe_unpaired = match_cohorts(cases, controls)

    weights = []
    for c in cases:
        weights.append(
            [
                abs((c.day - k.day).days) if abs((c.day - k.day).days) <= 7 else None
                for k in controls
            ]
        )

    best = None
    best_assignments = []
    # 每个病例：不配，或配一个对照；对照不重复。
    for assignment in itertools.product(range(-1, len(controls)), repeat=len(cases)):
        used = [j for j in assignment if j >= 0]
        if len(set(used)) != len(used):
            continue
        if any(assignment[i] >= 0 and weights[i][assignment[i]] is None for i in range(len(cases))):
            continue
        value = (len(used), -sum(weights[i][assignment[i]] for i in range(len(cases)) if assignment[i] >= 0))
        if best is None or value > best:
            best = value
            best_assignments = [assignment]
        elif value == best:
            best_assignments.append(assignment)

    assert len(pairs) == best[0]
    assert total == -best[1]
    for i in range(len(cases)):
        expect = {a[i] for a in best_assignments if a[i] >= 0}
        assert set(possible[i]) == expect
        assert maybe_unpaired[i] == any(a[i] < 0 for a in best_assignments)


# ---------------------------------------------------------------------------
# 队列构建与目标码解析
# ---------------------------------------------------------------------------

def test_cohorts_case_first_hit_control_earliest_visit():
    rows = [
        ("PSN-A", parse_visit_utc("2024-06-05T10:00:00Z"), "J18.9"),
        ("PSN-A", parse_visit_utc("2024-05-01T09:00:00Z"), "J18.9"),
        ("PSN-A", parse_visit_utc("2024-04-01T09:00:00Z"), "I10"),
        ("PSN-B", parse_visit_utc("2024-05-03T09:00:00Z"), "I10"),
        ("PSN-B", parse_visit_utc("2024-05-02T09:00:00Z"), "E11"),
    ]
    cases, controls = build_cohorts(rows, ["J18.9"])
    # 病例日期取首次命中（05-01），不是首次就诊（04-01）。
    assert [(c.code, c.day.isoformat()) for c in cases] == [("PSN-A", "2024-05-01")]
    # 对照日期取最早就诊日。
    assert [(c.code, c.day.isoformat()) for c in controls] == [("PSN-B", "2024-05-02")]


def test_cohorts_mixed_timestamp_formats():
    rows = [
        ("PSN-A", parse_visit_utc("2024-05-03T00:00:00Z"), "X"),
        ("PSN-A", parse_visit_utc("2024-05-02 23:59:59+00:00"), "X"),
    ]
    cases, _ = build_cohorts(rows, ["X"])
    assert [c.day.isoformat() for c in cases] == ["2024-05-02"]


def test_cohorts_merges_same_code_across_rows():
    rows = [
        ("PSN-A", parse_visit_utc("2024-05-01T00:00:00Z"), "I10"),
        ("PSN-A", parse_visit_utc("2024-06-01T00:00:00Z"), "J18.9"),
    ]
    cases, controls = build_cohorts(rows, ["J18.9"])
    assert [c.code for c in cases] == ["PSN-A"]
    assert controls == []


def test_parse_targets_split_and_dedupe():
    assert parse_targets(" J18.9, I10；J18.9、E11\nI10 ") == ["J18.9", "I10", "E11"]


def test_parse_targets_empty_rejected():
    with pytest.raises(AnalysisError):
        parse_targets("  , ")


def test_parse_targets_overlong_rejected():
    with pytest.raises(AnalysisError):
        parse_targets("D" * 101)


# ---------------------------------------------------------------------------
# Web 集成
# ---------------------------------------------------------------------------

@pytest.fixture()
def app(tmp_path):
    application = create_app({"DATA_DIR": str(tmp_path / "data")})
    application.config.update(TESTING=True)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


def post_csv(client, study, payload, filename="visits.csv"):
    return client.post(
        "/upload",
        data={"study": study, "file": (io.BytesIO(payload), filename)},
        content_type="multipart/form-data",
        follow_redirects=True,
    )


def post_analysis(client, study_id, batch_ids, targets):
    data = {"targets": targets, "batch_id": [str(b) for b in batch_ids]}
    if study_id is not None:
        data["study_id"] = str(study_id)
    return client.post("/analysis", data=data)


def codes_by_patient(app):
    db = sqlite3.connect(app.config["DATA_DIR"] + "/research.sqlite3")
    try:
        return dict(db.execute("SELECT patient_id, code FROM pseudonyms").fetchall())
    finally:
        db.close()


def test_analysis_page_and_nav(client):
    assert 'href="/analysis"' in client.get("/").data.decode()
    assert client.get("/analysis").status_code == 200
    post_csv(client, "随访研究", b"A,2024-05-01T09:00:00Z,X\n")
    page = client.get("/analysis?study_id=1")
    assert page.status_code == 200
    assert "批次 #1" in page.data.decode()


def test_analysis_end_to_end(client, app):
    post_csv(
        client,
        "随访研究",
        b"A,2024-05-01T09:00:00Z,J18.9\n"
        b"B,2024-05-02T09:00:00Z,I10\n"
        b"C,2024-05-10T09:00:00Z,E11.9\n",
    )
    post_csv(
        client,
        "随访研究",
        b"A,2024-06-01T09:00:00Z,J18.9\n"
        b"D,2024-06-03T09:00:00Z,K21.9\n"
        b"E,2024-05-03T09:00:00Z,J18.9\n",
    )
    response = post_analysis(client, 1, [1, 2], "J18.9")
    assert response.status_code == 200
    body = response.data.decode()
    assert "一组最优配对" in body
    assert "配对唯一性核查" in body
    assert "#1" in body and "#2" in body
    # 病例 A(05-01)、E(05-03)；对照 B(05-02)、C(05-10)、D(06-03)。
    # 唯一最优：A-B(1 天) + E-C(7 天)，总日差 8。
    assert "8 天" in body
    codes = codes_by_patient(app)
    for patient in ("A", "B", "C", "E"):
        assert codes[patient] in body
    # 两个病例都始终配给同一对照。
    assert body.count('badge-ok">确定') == 2
    assert 'badge-warn">不确定' not in body


def test_analysis_uncertain_and_maybe_unpaired(client):
    post_csv(
        client,
        "S",
        b"A,2024-05-01T09:00:00Z,X\n"
        b"B,2024-05-01T09:00:00Z,X\n"
        b"C,2024-05-01T09:00:00Z,Y\n"
        b"D,2024-05-01T09:00:00Z,Y\n"
        b"E,2024-05-01T09:00:00Z,X\n",
    )
    # 3 病例 2 对照，日期全同：每个病例都可能配到任一对照，也可能未配对。
    response = post_analysis(client, 1, [1], "X")
    body = response.data.decode()
    assert body.count('badge-warn">不确定') == 3
    assert body.count("<td>是</td>") == 3


def test_analysis_new_upload_not_mixed(client, app):
    post_csv(client, "S", b"A,2024-05-01T09:00:00Z,X\nB,2024-05-02T09:00:00Z,Y\n")
    post_csv(client, "S", b"C,2024-05-01T09:00:00Z,X\nD,2024-05-02T09:00:00Z,Y\n")
    codes = codes_by_patient(app)

    only_first = post_analysis(client, 1, [1], "X")
    body = only_first.data.decode()
    assert codes["A"] in body and codes["B"] in body
    assert codes["C"] not in body and codes["D"] not in body

    both = post_analysis(client, 1, [1, 2], "X")
    body = both.data.decode()
    assert codes["C"] in body and codes["D"] in body


def test_analysis_does_not_modify_data(client, app):
    post_csv(client, "S", b"A,2024-05-01T09:00:00Z,X\nB,2024-05-02T09:00:00Z,Y\n")
    data_dir = app.config["DATA_DIR"]
    db = sqlite3.connect(data_dir + "/research.sqlite3")
    before = (
        db.execute("SELECT COUNT(*) FROM batches").fetchone()[0],
        db.execute("SELECT COUNT(*) FROM pseudonyms").fetchone()[0],
        db.execute("SELECT COUNT(*) FROM studies").fetchone()[0],
    )
    db.close()
    files_before = {
        name: open(os.path.join(data_dir, "results", name), "rb").read()
        for name in os.listdir(os.path.join(data_dir, "results"))
    }
    response = post_analysis(client, 1, [1], "X")
    assert response.status_code == 200
    db = sqlite3.connect(data_dir + "/research.sqlite3")
    after = (
        db.execute("SELECT COUNT(*) FROM batches").fetchone()[0],
        db.execute("SELECT COUNT(*) FROM pseudonyms").fetchone()[0],
        db.execute("SELECT COUNT(*) FROM studies").fetchone()[0],
    )
    db.close()
    assert before == after
    assert files_before == {
        name: open(os.path.join(data_dir, "results", name), "rb").read()
        for name in os.listdir(os.path.join(data_dir, "results"))
    }


def test_analysis_row_limit_rejected(client):
    payload = b"".join(
        ("P%d,2024-05-01T09:00:00Z,X\n" % i).encode() for i in range(501)
    )
    post_csv(client, "S", payload)
    body = post_analysis(client, 1, [1], "X").data.decode()
    assert "超过 500 行上限" in body


def test_analysis_case_limit_rejected(client):
    payload = b"".join(
        ("P%d,2024-05-01T09:00:00Z,X\n" % i).encode() for i in range(13)
    )
    post_csv(client, "S", payload)
    body = post_analysis(client, 1, [1], "X").data.decode()
    assert "病例共 13 人，超过 12 人上限" in body


def test_analysis_control_limit_rejected(client):
    payload = b"A,2024-05-01T09:00:00Z,X\n" + b"".join(
        ("P%d,2024-05-01T09:00:00Z,Y\n" % i).encode() for i in range(13)
    )
    post_csv(client, "S", payload)
    body = post_analysis(client, 1, [1], "X").data.decode()
    assert "对照共 13 人，超过 12 人上限" in body


def test_analysis_cross_study_rejected(client):
    post_csv(client, "研究一", b"A,2024-05-01T09:00:00Z,X\n")
    post_csv(client, "研究二", b"B,2024-05-01T09:00:00Z,Y\n")
    body = post_analysis(client, 1, [2], "X").data.decode()
    assert "同一研究" in body


def test_analysis_input_errors(client):
    post_csv(client, "S", b"A,2024-05-01T09:00:00Z,X\n")
    assert "请填写目标诊断码" in post_analysis(client, 1, [1], "  ").data.decode()
    assert "请至少选择一个已发布批次" in post_analysis(client, 1, [], "X").data.decode()
    assert "批次 #999 不存在" in post_analysis(client, 1, [999], "X").data.decode()
    assert "请选择研究项目" in post_analysis(client, None, [1], "X").data.decode()


def test_analysis_with_header_row(client):
    post_csv(
        client,
        "S",
        b"patient_id,visit_utc,diagnosis_code\n"
        b"A,2024-05-01T09:00:00Z,X\n"
        b"B,2024-05-02T09:00:00Z,Y\n",
    )
    response = post_analysis(client, 1, [1], "X")
    body = response.data.decode()
    assert "一组最优配对" in body
    # 表头不计入明细行数（两行数据）。
    assert "分析明细行数</dt><dd>2</dd>" in body
