"""病例-对照匹配分析：只读已发布批次的替码结果文件。

流程
----
1. 读取所选批次的替码 CSV（只有登记过的批次才能被选中，分析不触碰
   替码登记表，也不写入任何文件，因此新上传的批次不会混入本次分析，
   原始数据也不会被改写）。
2. 按替码合并患者：任一明细行的诊断码命中目标码即为病例，病例日期为
   首次命中的 UTC 日期；从未命中者为对照，日期为最早就诊日。
3. 一对一配对：仅当病例与对照的日期相差不超过 7 天才允许配对，每人
   至多使用一次。目标按字典序优化——先最大化配对数，再最小化绝对
   日差总和。
4. 唯一性核查：用前向/后向动态规划表枚举“是否存在包含某条边的最优
   解”，逐病例给出所有最优方案中可能的对照以及是否可能未配对；
   只有始终配给同一对照的病例才标记为“确定”。

规模上限（超限即拒绝）：所选明细合计最多 500 行，病例、对照各最多
12 人。12 人上限使 2^12 的子集动态规划在请求线程内即可完成。
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from datetime import date

from . import store
from .validation import MAX_FIELD_CHARS, looks_like_header, parse_visit_utc

#: 所选明细（数据行，不含表头）合计上限。
MAX_ROWS = 500
#: 病例、对照各自的人数上限。
MAX_MEMBERS = 12
#: 目标诊断码个数上限。
MAX_TARGETS = 50
#: 允许配对的日期差（含边界）。
WINDOW_DAYS = 7


class AnalysisError(Exception):
    """输入或规模不满足要求；消息直接展示给用户。"""


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Member:
    """一名病例或对照：替码 + 用于配对的 UTC 日期。"""

    code: str
    day: date


@dataclass(frozen=True)
class PairRow:
    case: Member
    control: Member
    day_diff: int


@dataclass(frozen=True)
class CaseReport:
    """一个病例在所有最优方案中的配对情况。"""

    case: Member
    possible: tuple  # tuple[Member]，所有最优方案中出现过对照
    maybe_unpaired: bool  # 是否存在某个最优方案让该病例未配对
    certain: bool  # 始终配给同一对照


@dataclass(frozen=True)
class AnalysisResult:
    study_name: str
    batch_ids: tuple
    targets: tuple
    total_rows: int
    patient_count: int
    cases: tuple  # tuple[Member]
    controls: tuple  # tuple[Member]
    pairs: tuple  # tuple[PairRow]，一组最优配对
    total_day_diff: int
    unmatched_cases: tuple  # tuple[Member]
    unmatched_controls: tuple  # tuple[Member]
    case_reports: tuple  # tuple[CaseReport]


# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------

_TARGET_SPLIT_RE = re.compile(r"[,，;；、\s]+")


def parse_targets(raw: str | None) -> list[str]:
    """把用户输入拆成目标诊断码列表（精确匹配，区分大小写）。"""
    parts = [p for p in _TARGET_SPLIT_RE.split(raw or "") if p]
    targets: list[str] = []
    seen = set()
    for part in parts:
        if len(part) > MAX_FIELD_CHARS:
            raise AnalysisError(
                "目标诊断码超过 %d 个字符" % MAX_FIELD_CHARS
            )
        if part not in seen:
            seen.add(part)
            targets.append(part)
    if not targets:
        raise AnalysisError("请填写目标诊断码。")
    if len(targets) > MAX_TARGETS:
        raise AnalysisError("目标诊断码最多 %d 个。" % MAX_TARGETS)
    return targets


def _read_batch_rows(paths: store.Paths, result_file: str) -> list[tuple]:
    """读取一个已发布批次的替码 CSV，返回 (替码, 就诊时刻, 诊断码) 列表。"""
    full = os.path.join(paths.results_dir, result_file)
    rows: list[tuple] = []
    try:
        fh = open(full, "r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise AnalysisError("批次结果文件缺失或不可读。") from exc
    with fh:
        reader = csv.reader(fh)
        first = True
        for cells in reader:
            if not cells:
                continue
            if first and looks_like_header(cells):
                first = False
                continue
            first = False
            if len(cells) != 3:
                raise AnalysisError("结果文件 %s 内容异常：数据行不是 3 列。" % result_file)
            code, visit, diagnosis = cells
            try:
                when = parse_visit_utc(visit)
            except ValueError as exc:
                raise AnalysisError(
                    "结果文件 %s 内容异常：就诊时间非法。" % result_file
                ) from exc
            rows.append((code, when, diagnosis))
    return rows


def build_cohorts(rows: list[tuple], targets: list[str]) -> tuple[list[Member], list[Member]]:
    """按替码合并患者，分出病例与对照。

    病例：任一明细行诊断码命中目标码；日期为首次命中的 UTC 日期。
    对照：所选明细中从未命中；日期为最早就诊日。
    """
    target_set = set(targets)
    first_visit: dict[str, object] = {}
    first_hit: dict[str, object] = {}
    for code, when, diagnosis in rows:
        previous = first_visit.get(code)
        if previous is None or when < previous:
            first_visit[code] = when
        if diagnosis in target_set:
            hit = first_hit.get(code)
            if hit is None or when < hit:
                first_hit[code] = when
    cases: list[Member] = []
    controls: list[Member] = []
    for code, visit_min in first_visit.items():
        hit = first_hit.get(code)
        if hit is not None:
            cases.append(Member(code, hit.date()))
        else:
            controls.append(Member(code, visit_min.date()))
    # 排序保证结果与展示稳定（同人同日时按替码字典序）。
    cases.sort(key=lambda m: (m.day, m.code))
    controls.sort(key=lambda m: (m.day, m.code))
    return cases, controls


# ---------------------------------------------------------------------------
# 最优配对与唯一性核查
# ---------------------------------------------------------------------------

def _day_weights(cases: list[Member], controls: list[Member]) -> list[list]:
    """weights[i][j] = 日差（可配对）或 None（超过窗口）。"""
    weights: list[list] = []
    for case in cases:
        row = []
        for control in controls:
            diff = abs((case.day - control.day).days)
            row.append(diff if diff <= WINDOW_DAYS else None)
        weights.append(row)
    return weights


def _dp_tables(weights: list[list], n: int, m: int) -> tuple[list, list]:
    """前向/后向子集 DP。

    ``fwd[i][mask]`` 为前 i 个病例恰好使用 mask 中对照的最优值，
    ``bwd[i][mask]`` 为第 i 个及之后的病例恰好使用 mask 中对照的最优值。
    值为 (配对数, 负的日差总和)，按字典序取最大；None 表示不可达。
    """
    size = 1 << m
    fwd = [[None] * size for _ in range(n + 1)]
    fwd[0][0] = (0, 0)
    for i in range(n):
        cur, nxt = fwd[i], fwd[i + 1]
        w_row = weights[i]
        for mask in range(size):
            val = cur[mask]
            if val is None:
                continue
            if nxt[mask] is None or val > nxt[mask]:
                nxt[mask] = val
            for j in range(m):
                w = w_row[j]
                if w is None:
                    continue
                bit = 1 << j
                if mask & bit:
                    continue
                cand = (val[0] + 1, val[1] - w)
                merged = mask | bit
                if nxt[merged] is None or cand > nxt[merged]:
                    nxt[merged] = cand
    bwd = [[None] * size for _ in range(n + 1)]
    bwd[n][0] = (0, 0)
    for i in range(n - 1, -1, -1):
        nxt, cur = bwd[i + 1], bwd[i]
        w_row = weights[i]
        for mask in range(size):
            val = nxt[mask]
            if val is None:
                continue
            if cur[mask] is None or val > cur[mask]:
                cur[mask] = val
            for j in range(m):
                w = w_row[j]
                if w is None:
                    continue
                bit = 1 << j
                if mask & bit:
                    continue
                cand = (val[0] + 1, val[1] - w)
                merged = mask | bit
                if cur[merged] is None or cand > cur[merged]:
                    cur[merged] = cand
    return fwd, bwd


def _subset_maximize(table: list, m: int) -> list:
    """把“恰好使用 mask”的 DP 行转成“至多使用 mask”的上确界表。

    ``sup[i][U] = max{ table[i][S] : S ⊆ U }``，用 SOS（sum-over-subsets）
    动态规划求得，用于把前向/后向表组合成“其余病例”的最优值。
    """
    sup = [row[:] for row in table]
    size = 1 << m
    for row in sup:
        for j in range(m):
            bit = 1 << j
            for mask in range(size):
                if not mask & bit:
                    continue
                other = row[mask ^ bit]
                if other is not None and (row[mask] is None or other > row[mask]):
                    row[mask] = other
    return sup


def _extract_one_optimum(fwd: list, weights: list[list], n: int, m: int, optimum) -> list[tuple]:
    """沿前向 DP 回溯出一组最优配对（确定性）。"""
    size = 1 << m
    mask = -1
    for candidate in range(size):
        if fwd[n][candidate] == optimum:
            mask = candidate
            break
    pairs: list[tuple] = []
    for i in range(n - 1, -1, -1):
        value = fwd[i + 1][mask]
        # 优先选择“配对”分支：日差小、对照序号小者优先，保证输出稳定。
        options = sorted(
            (weights[i][j], j)
            for j in range(m)
            if weights[i][j] is not None and mask & (1 << j)
        )
        assigned = False
        for w, j in options:
            prev = fwd[i][mask ^ (1 << j)]
            if prev is not None and (prev[0] + 1, prev[1] - w) == value:
                pairs.append((i, j))
                mask ^= 1 << j
                assigned = True
                break
        if not assigned:
            # 该病例在此最优解中未配对（跳过分支达到同一最优值）。
            pass
    pairs.sort()
    return pairs


def match_cohorts(cases: list[Member], controls: list[Member]):
    """求最优配对并核查唯一性。

    返回 ``(pairs, unmatched_cases, unmatched_controls, total_day_diff,
    possible, maybe_unpaired)``，下标均为 cases/controls 中的序号：

    * ``pairs``：一组最优配对 [(病例序号, 对照序号)]；
    * ``possible[i]``：所有最优方案中病例 i 可能配到的对照序号列表；
    * ``maybe_unpaired[i]``：是否存在最优方案让病例 i 未配对。
    """
    n, m = len(cases), len(controls)
    weights = _day_weights(cases, controls)
    if n == 0 or m == 0:
        unmatched_cases = list(range(n))
        unmatched_controls = list(range(m))
        possible = [[] for _ in range(n)]
        maybe_unpaired = [True] * n
        return [], unmatched_cases, unmatched_controls, 0, possible, maybe_unpaired

    fwd, bwd = _dp_tables(weights, n, m)
    bwd_sup = _subset_maximize(bwd, m)
    full = (1 << m) - 1
    optimum = max(v for v in fwd[n] if v is not None)
    pairs = _extract_one_optimum(fwd, weights, n, m, optimum)
    total_day_diff = -optimum[1]

    def combine_best(i: int, allowed: int, extra) -> tuple | None:
        """病例 i 之外的最优值：前 i 个病例精确用 S，其余在 allowed\\S 内任选。"""
        best = None
        left_row = fwd[i]
        right_row = bwd_sup[i + 1]
        subset = allowed
        while True:
            left = left_row[subset]
            right = right_row[allowed ^ subset]
            if left is not None and right is not None:
                cand = (
                    left[0] + right[0] + extra[0],
                    left[1] + right[1] + extra[1],
                )
                if best is None or cand > best:
                    best = cand
            if subset == 0:
                break
            subset = (subset - 1) & allowed
        return best

    possible: list[list] = []
    maybe_unpaired: list[bool] = []
    for i in range(n):
        # 病例 i 不参与配对时，其余人的最优值是否仍等于全局最优。
        maybe_unpaired.append(combine_best(i, full, (0, 0)) == optimum)
        options = []
        for j in range(m):
            w = weights[i][j]
            if w is None:
                continue
            # 强制 (i, j) 配对后，剩余子问题的最优值加上这条边。
            forced = combine_best(i, full ^ (1 << j), (1, -w))
            if forced == optimum:
                options.append(j)
        possible.append(options)

    paired_cases = {i for i, _ in pairs}
    paired_controls = {j for _, j in pairs}
    unmatched_cases = [i for i in range(n) if i not in paired_cases]
    unmatched_controls = [j for j in range(m) if j not in paired_controls]
    return pairs, unmatched_cases, unmatched_controls, total_day_diff, possible, maybe_unpaired


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------

def run_analysis(
    conn,
    paths: store.Paths,
    study_id: int | None,
    batch_ids: list[int],
    raw_targets: str | None,
) -> AnalysisResult:
    """校验输入、读取所选批次并完成匹配分析。

    只读操作：不写入数据库或文件系统。任何输入/规模问题都以
    :class:`AnalysisError` 抛出。
    """
    targets = parse_targets(raw_targets)
    if study_id is None:
        raise AnalysisError("请选择研究项目。")
    study = store.fetch_study(conn, study_id)
    if study is None:
        raise AnalysisError("研究项目不存在。")

    unique_ids: list[int] = []
    seen = set()
    for batch_id in batch_ids or []:
        if batch_id not in seen:
            seen.add(batch_id)
            unique_ids.append(batch_id)
    if not unique_ids:
        raise AnalysisError("请至少选择一个已发布批次。")
    if len(unique_ids) > MAX_ROWS:
        raise AnalysisError("一次最多选择 %d 个批次。" % MAX_ROWS)

    batches = []
    declared_rows = 0
    for batch_id in unique_ids:
        batch = store.fetch_batch(conn, batch_id)
        if batch is None:
            raise AnalysisError("批次 #%d 不存在。" % batch_id)
        if batch.study_id != study_id:
            raise AnalysisError("只能选择同一研究项目的批次。")
        batches.append(batch)
        declared_rows += batch.rows
    if declared_rows > MAX_ROWS:
        raise AnalysisError(
            "所选明细共 %d 行，超过 %d 行上限，已拒绝。" % (declared_rows, MAX_ROWS)
        )

    rows: list[tuple] = []
    for batch in batches:
        rows.extend(_read_batch_rows(paths, batch.result_file))
        if len(rows) > MAX_ROWS:
            raise AnalysisError(
                "所选明细超过 %d 行上限，已拒绝。" % MAX_ROWS
            )

    cases, controls = build_cohorts(rows, targets)
    if len(cases) > MAX_MEMBERS:
        raise AnalysisError(
            "病例共 %d 人，超过 %d 人上限，已拒绝。" % (len(cases), MAX_MEMBERS)
        )
    if len(controls) > MAX_MEMBERS:
        raise AnalysisError(
            "对照共 %d 人，超过 %d 人上限，已拒绝。" % (len(controls), MAX_MEMBERS)
        )

    pairs, unmatched_cases, unmatched_controls, total_day_diff, possible, maybe_unpaired = (
        match_cohorts(cases, controls)
    )

    pair_rows = tuple(
        PairRow(cases[i], controls[j], abs((cases[i].day - controls[j].day).days))
        for i, j in pairs
    )
    case_reports = tuple(
        CaseReport(
            case=cases[i],
            possible=tuple(controls[j] for j in possible[i]),
            maybe_unpaired=maybe_unpaired[i],
            certain=(len(possible[i]) == 1 and not maybe_unpaired[i]),
        )
        for i in range(len(cases))
    )
    return AnalysisResult(
        study_name=study["name"],
        batch_ids=tuple(unique_ids),
        targets=tuple(targets),
        total_rows=len(rows),
        patient_count=len(cases) + len(controls),
        cases=tuple(cases),
        controls=tuple(controls),
        pairs=pair_rows,
        total_day_diff=total_day_diff,
        unmatched_cases=tuple(cases[i] for i in unmatched_cases),
        unmatched_controls=tuple(controls[j] for j in unmatched_controls),
        case_reports=case_reports,
    )
