"""Flask application: upload form, batch registry and result downloads."""

from __future__ import annotations

import logging
import os
import sqlite3

from flask import (
    Flask,
    abort,
    flash,
    g as _ctx,
    get_flashed_messages,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge

from . import store
from .analysis import AnalysisError, run_analysis
from .pipeline import convert_upload
from .store import ConflictError, StorageError
from .validation import (
    MAX_FILE_BYTES,
    MAX_FIELD_CHARS,
    FileTooLarge,
    InvalidBatch,
    InvalidStudyName,
)

log = logging.getLogger("research.web")

#: Multipart overhead allowance on top of the CSV body limit.
FORM_OVERHEAD = 8 * 1024 * 1024

ALLOWED_EXTENSIONS = (".csv",)


def _fmt_bytes(value: int | None) -> str:
    if not value:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return "%.1f %s" % (size, unit) if unit != "B" else "%d B" % size
        size /= 1024
    return "%d B" % value


def create_app(settings: dict | None = None) -> Flask:
    settings = dict(settings or {})
    data_dir = settings.get("DATA_DIR") or os.environ.get("DATA_DIR", "/data")
    max_upload = int(
        settings.get("MAX_UPLOAD_BYTES")
        or os.environ.get("MAX_UPLOAD_BYTES", MAX_FILE_BYTES)
    )
    # Never allow the configured limit to exceed the documented 256 MiB.
    max_upload = min(max_upload, MAX_FILE_BYTES)

    paths = store.Paths.build(data_dir)
    store.initialize(paths)
    secret = store.load_secret(paths)

    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=max_upload + FORM_OVERHEAD,
        MAX_UPLOAD_BYTES=max_upload,
        DATA_DIR=paths.data_dir,
        JSON_SORT_KEYS=False,
        SECRET_KEY=store.load_flash_secret(paths),
    )
    app.jinja_env.filters["bytes"] = _fmt_bytes
    app.jinja_env.globals.update(
        MAX_FIELD_CHARS=MAX_FIELD_CHARS,
        MAX_UPLOAD_BYTES=max_upload,
    )

    # Startup recovery: drop unregistered result files and stale work dirs.
    cleanup_conn = store.connect(paths.db_path)
    try:
        swept = store.cleanup_unpublished(paths, cleanup_conn)
    finally:
        cleanup_conn.close()
    if swept["result_files"] or swept["tmp_entries"]:
        log.warning(
            "startup cleanup removed %d unpublished result file(s) and %d temp entry(s)",
            len(swept["result_files"]),
            swept["tmp_entries"],
        )

    def db() -> sqlite3.Connection:
        conn = getattr(_ctx, "conn", None)
        if conn is None:
            conn = store.connect(paths.db_path)
            _ctx.conn = conn
        return conn

    @app.teardown_appcontext
    def _close_db(exception=None):
        conn = getattr(_ctx, "conn", None)
        if conn is not None:
            conn.close()
            _ctx.conn = None

    @app.get("/")
    def index():
        conn = db()
        return render_template(
            "index.html",
            studies=store.list_studies(conn),
            batches=store.list_batches(conn, limit=20),
            messages=get_flashed_messages(with_categories=True),
            study_value=request.args.get("study", ""),
        )

    @app.post("/upload")
    def upload():
        raw_study = request.form.get("study", "")
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            flash("请选择要上传的 CSV 文件。", "error")
            return redirect(url_for("index"))
        if not uploaded.filename.lower().endswith(ALLOWED_EXTENSIONS):
            flash("仅接受 .csv 文件。", "error")
            return redirect(url_for("index"))

        conn = db()
        stream = uploaded.stream
        stream.seek(0, os.SEEK_END)
        declared = stream.tell()
        stream.seek(0)
        if declared > max_upload:
            flash(
                "文件大小 %d 字节（%s）超过上限 %d 字节（%s），整批已拒绝。"
                % (declared, _fmt_bytes(declared), max_upload, _fmt_bytes(max_upload)),
                "error",
            )
            return redirect(url_for("index"))

        try:
            result = convert_upload(
                conn,
                paths,
                secret,
                raw_study,
                stream,
                max_bytes=max_upload,
            )
        except InvalidStudyName as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))
        except FileTooLarge as exc:
            flash(
                "文件超过上限 %d 字节（%s），整批已拒绝。"
                % (exc.limit, _fmt_bytes(exc.limit)),
                "error",
            )
            return redirect(url_for("index"))
        except InvalidBatch as exc:
            flash(
                "整批已拒绝：第 %d 行 %s。本次未生成替码，也没有可下载结果。"
                % (exc.line_no, exc.message),
                "error",
            )
            return redirect(url_for("index"))
        except ConflictError as exc:
            flash("整批已拒绝：%s。" % exc, "error")
            return redirect(url_for("index"))
        except StorageError as exc:
            log.exception("storage failure while publishing batch")
            flash(
                "整批已拒绝：结果写入或登记失败（%s）。未留下替码或可下载结果。"
                % exc,
                "error",
            )
            return redirect(url_for("index"))

        log.info(
            "batch %s published study=%r rows=%d new=%d reused=%d",
            result.batch_id,
            result.study,
            result.stats.rows,
            result.stats.new_codes,
            result.stats.reused_rows,
        )
        flash(
            "已生成替码文件：%d 行，新增替码 %d 个，复用 %d 行。"
            % (result.stats.rows, result.stats.new_codes, result.stats.reused_rows),
            "ok",
        )
        return redirect(url_for("batch_detail", batch_id=result.batch_id))

    @app.get("/batches/<int:batch_id>")
    def batch_detail(batch_id: int):
        conn = db()
        batch = store.fetch_batch(conn, batch_id)
        if batch is None:
            abort(404)
        study = store.fetch_study(conn, batch.study_id)
        return render_template(
            "batch.html",
            batch=batch,
            study=study,
            messages=get_flashed_messages(with_categories=True),
        )

    @app.get("/batches")
    def batches():
        conn = db()
        study_id = request.args.get("study_id", type=int)
        return render_template(
            "batches.html",
            batches=store.list_batches(conn, study_id=study_id, limit=200),
            studies=store.list_studies(conn),
            selected=study_id,
            messages=get_flashed_messages(with_categories=True),
        )

    @app.get("/studies/<int:study_id>")
    def study_detail(study_id: int):
        conn = db()
        study = store.fetch_study(conn, study_id)
        if study is None:
            abort(404)
        return render_template(
            "study.html",
            study=study,
            batches=store.list_batches(conn, study_id=study_id, limit=200),
            messages=get_flashed_messages(with_categories=True),
        )

    @app.get("/results/<name>")
    def download(name: str):
        if not store.RESULT_NAME_RE.match(name):
            abort(404)
        conn = db()
        row = conn.execute(
            "SELECT batch_id FROM batches WHERE result_file = ?", (name,)
        ).fetchone()
        if row is None:
            abort(404)
        path = os.path.join(paths.results_dir, name)
        if not os.path.isfile(path):
            abort(404)
        download_name = "pseudonymised-batch-%d.csv" % row[0]
        return send_file(
            path,
            mimetype="text/csv; charset=utf-8",
            as_attachment=True,
            download_name=download_name,
            conditional=True,
            max_age=0,
        )

    @app.route("/analysis", methods=["GET", "POST"])
    def analysis():
        conn = db()
        studies = store.list_studies(conn)
        result = None
        error = None
        checked: set[int] = set()
        targets_value = ""
        if request.method == "POST":
            study_id = request.form.get("study_id", type=int)
            batch_ids = request.form.getlist("batch_id", type=int)
            targets_value = request.form.get("targets", "")
            checked = set(batch_ids)
            try:
                result = run_analysis(conn, paths, study_id, batch_ids, targets_value)
            except AnalysisError as exc:
                error = str(exc)
        else:
            study_id = request.args.get("study_id", type=int)
        batches = (
            store.list_batches(conn, study_id=study_id, limit=500)
            if study_id is not None
            else []
        )
        return render_template(
            "analysis.html",
            studies=studies,
            selected_study=study_id,
            batches=batches,
            checked=checked,
            targets_value=targets_value,
            result=result,
            error=error,
            messages=get_flashed_messages(with_categories=True),
        )

    @app.get("/healthz")
    def healthz():
        try:
            ready = store.database_ready(db())
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            return {"ok": False, "error": str(exc)}, 503
        if not ready:
            return {"ok": False, "error": "database not initialised"}, 503
        return {"ok": True, "data_dir": paths.data_dir}

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        flash(
            "上传请求超过服务器上限，整批已拒绝。",
            "error",
        )
        return redirect(url_for("index"))

    @app.errorhandler(404)
    def not_found(_exc):
        return render_template("error.html", code=404, message="页面或资源不存在"), 404

    return app


def main() -> None:  # pragma: no cover - entry point used by the container
    from waitress import serve

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = create_app()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    log.info("serving on %s:%d data_dir=%s", host, port, app.config["DATA_DIR"])
    serve(
        app,
        host=host,
        port=port,
        threads=int(os.environ.get("THREADS", "16")),
        connection_limit=200,
        channel_timeout=900,
        max_request_body_size=app.config["MAX_CONTENT_LENGTH"] + 1024,
        ident="research-pseudonymiser",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
