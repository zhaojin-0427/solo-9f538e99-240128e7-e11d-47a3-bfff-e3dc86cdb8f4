"""Streaming validation and pseudonym substitution for visit-detail CSV files.

Invariants enforced on every row (any violation rejects the whole batch):

* exactly three columns: patient id, visit UTC timestamp, diagnosis code;
* no field is empty or whitespace-only;
* every field is at most :data:`MAX_FIELD_CHARS` Unicode characters;
* the visit time is a real calendar instant expressed in UTC
  (``Z`` / ``UTC`` / ``+00:00`` suffix required);
* input is UTF-8 (optional BOM), one record per physical line.

The input is processed line-by-line so memory use stays bounded regardless of
file size.  Output preserves every value except the patient id, which is
replaced by the assigned pseudonym, and preserves row order and line endings.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re

MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_FIELD_CHARS = 100

#: A legal record is at most three quoted 100-character fields.  A 100-character
#: UTF-8 field can be 300 bytes, so 4 KiB would already suffice; 8 KiB leaves
#: slack while keeping per-request memory bounded under hostile input.
MAX_LINE_BYTES = 8 * 1024

BOM_BYTES = b"\xef\xbb\xbf"
BOM_CHAR = "﻿"

csv.field_size_limit(MAX_LINE_BYTES)


class InvalidBatch(Exception):
    """The upload must be rejected; nothing may be published."""

    def __init__(self, line_no: int, message: str):
        super().__init__(message)
        self.line_no = line_no
        self.message = message


class FileTooLarge(Exception):
    def __init__(self, limit: int):
        super().__init__("文件超过 %d 字节上限" % limit)
        self.limit = limit


class InvalidStudyName(Exception):
    pass


# ---------------------------------------------------------------------------
# Study names
# ---------------------------------------------------------------------------

MAX_STUDY_NAME_CHARS = 100


def normalize_study_name(raw: str | None) -> str:
    """Trim, NFC-normalise and collapse internal whitespace in a study name."""
    import unicodedata

    text = unicodedata.normalize("NFC", raw or "").strip()
    text = " ".join(text.split())
    if not text:
        raise InvalidStudyName("研究项目不能为空")
    if len(text) > MAX_STUDY_NAME_CHARS:
        raise InvalidStudyName("研究项目名称不能超过 %d 个字符" % MAX_STUDY_NAME_CHARS)
    for ch in text:
        if ord(ch) < 0x20 or ord(ch) == 0x7F:
            raise InvalidStudyName("研究项目名称包含控制字符")
    return text


# ---------------------------------------------------------------------------
# Visit timestamps
# ---------------------------------------------------------------------------

_UTC_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})[T ]"
    r"(?P<H>\d{2}):(?P<M>\d{2})"
    r"(?::(?P<S>\d{2})(?:\.(?P<f>\d{1,6}))?)?"
    r"(?P<tz>Z|UTC|[+-]00:00)$",
    re.IGNORECASE,
)

_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _days_in_month(year: int, month: int) -> int:
    if month == 2 and ((year % 4 == 0 and year % 100 != 0) or year % 400 == 0):
        return 29
    return _MONTH_DAYS[month - 1]


def parse_visit_utc(text: str):
    """Parse a UTC timestamp into an aware ``datetime``.

    Accepts exactly the forms :data:`_UTC_RE` allows (``Z`` / ``UTC`` /
    ``+00:00`` suffix); every result carries ``tzinfo=utc`` so instants
    compare correctly regardless of the original spelling.
    """
    from datetime import datetime, timezone

    m = _UTC_RE.match(text)
    if not m:
        raise ValueError("需要 ISO-8601 UTC 时间，例如 2024-05-01T09:30:00Z")
    year, month, day = int(m["y"]), int(m["m"]), int(m["d"])
    hour, minute = int(m["H"]), int(m["M"])
    second = int(m["S"]) if m["S"] is not None else 0
    if year < 1:
        raise ValueError("年份超出范围")
    if not 1 <= month <= 12:
        raise ValueError("月份超出范围")
    if not 1 <= day <= _days_in_month(year, month):
        raise ValueError("日期不存在")
    if hour > 23 or minute > 59 or second > 59:
        raise ValueError("时间超出范围")
    micros = int((m["f"] or "0").ljust(6, "0"))
    return datetime(
        year, month, day, hour, minute, second, micros, tzinfo=timezone.utc
    )


def validate_visit_utc(text: str) -> str:
    """Return *text* unchanged when it denotes a valid UTC instant."""
    parse_visit_utc(text)
    return text


# ---------------------------------------------------------------------------
# Field checks
# ---------------------------------------------------------------------------

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _check_field(line_no: int, label: str, value: str) -> None:
    if not value or value.isspace():
        raise InvalidBatch(line_no, "%s缺失" % label)
    if len(value) > MAX_FIELD_CHARS:
        raise InvalidBatch(line_no, "%s超过 %d 个字符" % (label, MAX_FIELD_CHARS))
    if _CONTROL_RE.search(value) is not None:
        raise InvalidBatch(line_no, "%s包含控制字符" % label)


# ---------------------------------------------------------------------------
# Optional header row
# ---------------------------------------------------------------------------

def _norm_header(cell: str) -> str:
    import unicodedata

    cell = unicodedata.normalize("NFC", cell).strip().casefold()
    return re.sub(r"[\s_\-]+", "", cell)


_PATIENT_HEADERS = {
    "patientid", "patient", "patientno", "patientnumber", "患者号", "患者编号",
    "病人号", "病历号", "患者id",
}
_VISIT_HEADERS = {
    "visitutc", "visittime", "visitutctime", "就诊utc时间", "就诊时间",
    "就诊日期时间", "就诊utc",
}
_DIAG_HEADERS = {
    "diagnosiscode", "diagnosis", "诊断码", "诊断编码", "诊断代码",
}


def _looks_like_header(cells: list[str]) -> bool:
    return (
        len(cells) == 3
        and _norm_header(cells[0]) in _PATIENT_HEADERS
        and _norm_header(cells[1]) in _VISIT_HEADERS
        and _norm_header(cells[2]) in _DIAG_HEADERS
    )


#: Public alias so the analysis reader can skip the optional header row that
#: the converter passes through unchanged.
looks_like_header = _looks_like_header


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _escape(value: str) -> str:
    if "," in value or '"' in value or "\r" in value or "\n" in value:
        return '"' + value.replace('"', '""') + '"'
    return value


def _render(cells: list[str], newline: str) -> bytes:
    return (",".join(_escape(c) for c in cells) + newline).encode("utf-8")


def _first_field_span(body: str) -> int | None:
    """Return the index just past the first CSV field, or ``None``.

    Used to substitute the patient id without rewriting the remaining
    bytes of the record, so visit time and diagnosis stay byte-identical.
    """
    if not body:
        return None
    if body[0] == '"':
        i = 1
        n = len(body)
        while i < n:
            if body[i] == '"':
                if i + 1 < n and body[i + 1] == '"':
                    i += 2
                    continue
                break
            i += 1
        else:
            return None
        if i >= n:
            return None
        if i + 1 < n and body[i + 1] != ",":
            return None
        return i + 1
    comma = body.find(",")
    return len(body) if comma < 0 else comma


def _substitute_first_field(body: str, cells: list[str], code: str) -> bytes | None:
    """Replace only the first field's raw text, keeping the rest intact."""
    if body[0] != '"' and '"' not in body and "," in body:
        # Fast path: plain unquoted record, so the first comma ends field 1.
        comma = body.find(",")
        return (code + body[comma:]).encode("utf-8")
    end = _first_field_span(body)
    if end is None:
        return None
    prefix = body[:end]
    rest = body[end:]
    try:
        parsed = next(csv.reader([prefix], strict=True))
    except (csv.Error, StopIteration):
        return None
    if parsed != [cells[0]]:
        return None
    if rest and not rest.startswith(","):
        return None
    return (code + rest).encode("utf-8")


def _split_terminator(text: str) -> tuple[str, str]:
    if text.endswith("\r\n"):
        return text[:-2], "\r\n"
    if text.endswith("\n"):
        return text[:-1], "\n"
    return text, ""


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

class ConversionStats:
    __slots__ = (
        "rows", "new_codes", "reused_rows", "bytes_in", "bytes_out", "sha256",
        "had_header",
    )

    def __init__(self):
        self.rows = 0
        self.new_codes = 0
        self.reused_rows = 0
        self.bytes_in = 0
        self.bytes_out = 0
        self.sha256 = ""
        self.had_header = False

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "new_codes": self.new_codes,
            "reused_rows": self.reused_rows,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "sha256": self.sha256,
            "had_header": self.had_header,
        }


def convert_csv(
    stream,
    output_path: str,
    resolve_code,
    max_bytes: int = MAX_FILE_BYTES,
) -> ConversionStats:
    """Validate *stream* and write the masked CSV to *output_path*.

    ``resolve_code(patient_id) -> code`` assigns or reuses a pseudonym.  The caller must discard both the output file and any staged
    pseudonyms if this function raises.
    """
    stats = ConversionStats()
    hasher = hashlib.sha256()
    consumed = 0
    line_no = 0
    saw_any = False
    out = open(output_path, "wb", buffering=256 * 1024)
    try:
        while True:
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if raw == b"":
                break
            line_no += 1
            consumed += len(raw)
            if consumed > max_bytes:
                raise FileTooLarge(max_bytes)
            if len(raw) > MAX_LINE_BYTES:
                raise InvalidBatch(
                    line_no, "单行超过 %d 字节，字段超出允许长度" % MAX_LINE_BYTES
                )
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise InvalidBatch(line_no, "不是有效的 UTF-8 编码") from exc

            has_bom = line_no == 1 and text.startswith(BOM_CHAR)
            if has_bom:
                text = text[1:]
            body, terminator = _split_terminator(text)

            if body.strip() == "":
                raise InvalidBatch(line_no, "存在空行")
            try:
                cells = next(csv.reader([body], strict=True))
            except csv.Error as exc:
                raise InvalidBatch(line_no, "CSV 格式错误：%s" % exc) from exc
            except StopIteration:  # pragma: no cover - csv never yields empty
                cells = []

            if not saw_any and _looks_like_header(cells):
                stats.had_header = True
                saw_any = True
                # Pass the header through byte-for-byte.
                chunk = body.encode("utf-8") + (terminator or "\n").encode("utf-8")
                if has_bom:
                    chunk = BOM_BYTES + chunk
                out.write(chunk)
                hasher.update(chunk)
                continue
            saw_any = True

            if len(cells) != 3:
                raise InvalidBatch(
                    line_no, "需要 3 列，实际为 %d 列" % len(cells)
                )
            patient, visit, diagnosis = cells
            if not patient or patient.isspace():
                raise InvalidBatch(line_no, "患者号缺失")
            if patient != patient.strip():
                # " P1" and "P1" must not silently become two patients.
                raise InvalidBatch(line_no, "患者号首尾存在空白字符")
            if not visit or visit.isspace():
                raise InvalidBatch(line_no, "就诊 UTC 时间缺失")
            if not diagnosis or diagnosis.isspace():
                raise InvalidBatch(line_no, "诊断码缺失")
            if len(patient) > MAX_FIELD_CHARS:
                raise InvalidBatch(line_no, "患者号超过 %d 个字符" % MAX_FIELD_CHARS)
            if len(diagnosis) > MAX_FIELD_CHARS:
                raise InvalidBatch(line_no, "诊断码超过 %d 个字符" % MAX_FIELD_CHARS)
            if _CONTROL_RE.search(patient) or _CONTROL_RE.search(diagnosis):
                raise InvalidBatch(line_no, "字段包含控制字符")
            try:
                validate_visit_utc(visit)
            except ValueError as exc:
                raise InvalidBatch(
                    line_no, "就诊 UTC 时间非法：%s" % exc
                ) from exc
            if len(visit) > MAX_FIELD_CHARS:
                raise InvalidBatch(
                    line_no, "就诊 UTC 时间超过 %d 个字符" % MAX_FIELD_CHARS
                )
            if _CONTROL_RE.search(visit):
                raise InvalidBatch(line_no, "就诊 UTC 时间包含控制字符")

            code = resolve_code(patient)
            stats.rows += 1
            chunk = _substitute_first_field(body, cells, code)
            if chunk is None:
                # Unusual quoting around the patient id; rebuild the record
                # from the parsed values (the other two columns are unchanged).
                chunk = _render([code, cells[1], cells[2]], "")
            if terminator:
                chunk += terminator.encode("utf-8")
            if has_bom:
                chunk = BOM_BYTES + chunk
            out.write(chunk)
            hasher.update(chunk)

        if stats.rows == 0:
            raise InvalidBatch(1, "文件中没有可转换的数据行")
        stats.bytes_in = consumed
        stats.sha256 = hasher.hexdigest()
        out.flush()
        os.fsync(out.fileno())
        stats.bytes_out = out.tell()
        return stats
    finally:
        out.close()
