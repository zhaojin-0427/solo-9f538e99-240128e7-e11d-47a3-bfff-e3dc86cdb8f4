"""End-to-end conversion of one upload: validate, mask, publish.

The pipeline runs entirely in the request thread and takes no global lock, so
an upload for another study can be processed while a large file is being
converted.  The only serialised step is the short publication transaction in
:func:`store.commit_batch`.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections import OrderedDict

from . import store
from .store import StorageError
from .validation import (
    ConversionStats,
    InvalidBatch,
    convert_csv,
    normalize_study_name,
)

log = logging.getLogger("research.pipeline")


class ConversionResult:
    __slots__ = ("batch_id", "study", "stats", "result_file")

    def __init__(self, batch_id: int, study: str, stats: ConversionStats, result_file: str):
        self.batch_id = batch_id
        self.study = study
        self.stats = stats
        self.result_file = result_file


def convert_upload(
    conn: sqlite3.Connection,
    paths: store.Paths,
    secret: bytes,
    raw_study: str,
    stream,
    max_bytes: int,
) -> ConversionResult:
    """Validate, pseudonymise and publish one upload.

    Raises :class:`InvalidBatch`, :class:`FileTooLarge`, :class:`StorageError`
    or :class:`ConflictError`; in every failure case no result file, batch row
    or pseudonym survives.
    """
    study = normalize_study_name(raw_study)
    staging = store.Staging(paths)
    result_file: str | None = None
    published = False
    try:
        # A small bounded cache only; the staging database is the source of
        # truth for within-batch duplicates, so memory stays flat even for a
        # file with millions of distinct patients.
        memo: OrderedDict[str, str] = OrderedDict()
        memo_limit = 20000
        derive = store.code_hasher(secret, study)

        def remember(patient: str, code: str) -> None:
            memo[patient] = code
            memo.move_to_end(patient)
            if len(memo) > memo_limit:
                memo.popitem(last=False)

        def resolve(patient: str) -> str:
            # Codes are derived deterministically, so no registry read is
            # needed on the hot path: a patient registered earlier derives
            # the same code, and the publication transaction verifies every
            # (study, patient, code) pair before committing.
            cached = memo.get(patient)
            if cached is not None:
                memo.move_to_end(patient)
                return cached
            code = derive(patient)
            staging.add(patient, code)
            remember(patient, code)
            return code

        started = time.monotonic()
        stats = convert_csv(stream, staging.work_csv, resolve, max_bytes=max_bytes)
        converted = time.monotonic()
        # The file only becomes visible together with its registry row: the
        # lock keeps start-up cleanup in other processes from deleting a file
        # between these two steps.
        with store.publish_lock(paths):
            result_file = store.save_result_file(paths, staging.work_csv)
            batch_id, inserted = store.commit_batch(
                conn,
                staging,
                result_file,
                study,
                rows=stats.rows,
                bytes_in=stats.bytes_in,
                bytes_out=stats.bytes_out,
                sha256=stats.sha256,
            )
        committed = time.monotonic()
        stats.new_codes = inserted
        stats.reused_rows = max(stats.rows - inserted, 0)
        published = True
        log.info(
            "batch=%s study=%r rows=%d convert=%.1fs publish=%.1fs total=%.1fs",
            batch_id,
            study,
            stats.rows,
            converted - started,
            committed - converted,
            committed - started,
        )
        return ConversionResult(batch_id, study, stats, result_file)
    except (InvalidBatch, store.ConflictError):
        raise
    except sqlite3.Error as exc:
        raise StorageError("登记失败：%s" % exc) from exc
    except OSError as exc:
        raise StorageError("写入结果文件失败：%s" % exc) from exc
    finally:
        # Roll back everything that did not reach the publication commit.  A
        # file that is already registered is kept even if the response failed.
        if not published and result_file is not None and not store.is_registered(conn, result_file):
            store.remove_result_file(paths, result_file)
        staging.discard()
