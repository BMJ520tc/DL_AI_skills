"""知识库读写服务：统一索引、检索与四类数据入库接口（模块详细设计 2.6，数据设计二/三/十，D9）。

入库与索引在同一事务同步写，保证「先落库后检索」（数据设计一.2）。
unified_index 的 FTS 同步由 schema.sql 触发器自动维护。
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.db.connection import get_connection

# data_type -> (主表, 主键列)
_TABLE_PK = {
    "paper": ("paper", "paper_id"),
    "run": ("run_record", "run_id"),
    "knowledge": ("knowledge", "knowledge_id"),
    "dataset": ("dataset_registry", "dataset_id"),
    "module": ("module", "module_id"),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fts_query(q: str) -> str:
    """把用户输入转成 FTS5 短语查询，避免查询语法注入。"""
    return '"' + q.replace('"', '""') + '"'


def index_entry(
    conn,
    data_type: str,
    ref_id: str,
    *,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    source_project_id: Optional[str] = None,
    task_type: Optional[str] = None,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    tags: Optional[list] = None,
    keywords: Optional[str] = None,
) -> None:
    """写/更新统一索引一条（UPSERT，需在调用方事务内）。"""
    index_id = f"{data_type}:{ref_id}"
    now = _now()
    conn.execute(
        """
        INSERT INTO unified_index(id, data_type, ref_id, title, summary, source_project_id,
            task_type, model_name, dataset_name, tags, keywords, schema_version, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '1.0', ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title, summary = excluded.summary,
            source_project_id = excluded.source_project_id, task_type = excluded.task_type,
            model_name = excluded.model_name, dataset_name = excluded.dataset_name,
            tags = excluded.tags, keywords = excluded.keywords, updated_at = excluded.updated_at
        """,
        (
            index_id,
            data_type,
            ref_id,
            title,
            summary,
            source_project_id,
            task_type,
            model_name,
            dataset_name,
            json.dumps(tags, ensure_ascii=False) if tags is not None else None,
            keywords,
            now,
            now,
        ),
    )


def record_run(run: dict) -> str:
    """写入运行记录（run_record）并同步统一索引（数据设计五.4）。"""
    run_id = run.get("run_id") or uuid.uuid4().hex
    duration_s = run.get("duration_s")
    if duration_s is None and run.get("started_at") and run.get("finished_at"):
        try:
            start = datetime.fromisoformat(run["started_at"])
            finish = datetime.fromisoformat(run["finished_at"])
            duration_s = (finish - start).total_seconds()
        except (ValueError, TypeError):
            duration_s = None
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO run_record(run_id, project_id, task_id, run_type, environment, params,
                command, status, metrics, error, artifact_path, log_path, started_at, finished_at,
                duration_s, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '1.0')
            """,
            (
                run_id,
                run.get("project_id"),
                run.get("task_id"),
                run.get("run_type"),
                json.dumps(run.get("environment"), ensure_ascii=False) if run.get("environment") else None,
                json.dumps(run.get("params"), ensure_ascii=False) if run.get("params") else None,
                run.get("command"),
                run.get("status", "running"),
                json.dumps(run.get("metrics"), ensure_ascii=False) if run.get("metrics") else None,
                run.get("error"),
                run.get("artifact_path"),
                run.get("log_path"),
                run.get("started_at"),
                run.get("finished_at"),
                duration_s,
            ),
        )
        index_entry(
            conn,
            "run",
            run_id,
            title=run.get("run_type"),
            summary=run.get("command") or run.get("error"),
            source_project_id=run.get("project_id"),
            tags=[run.get("run_type")] if run.get("run_type") else None,
            keywords=run.get("run_type"),
        )
        conn.commit()
    finally:
        conn.close()
    return run_id


def record_paper(paper: dict) -> str:
    """写入论文记录并同步统一索引（数据设计四.3）。"""
    paper_id = paper.get("paper_id") or uuid.uuid4().hex
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO paper(paper_id, title, authors, abstract, source, url, published_date,
                license, pdf_path, markdown_path, section_index, status, created_at, updated_at, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '1.0')
            """,
            (
                paper_id,
                paper.get("title"),
                json.dumps(paper.get("authors"), ensure_ascii=False) if paper.get("authors") else None,
                paper.get("abstract"),
                paper.get("source"),
                paper.get("url"),
                paper.get("published_date"),
                paper.get("license"),
                paper.get("pdf_path"),
                paper.get("markdown_path"),
                json.dumps(paper.get("section_index"), ensure_ascii=False) if paper.get("section_index") else None,
                paper.get("status", "downloaded"),
                _now(),
                _now(),
            ),
        )
        index_entry(
            conn,
            "paper",
            paper_id,
            title=paper.get("title"),
            summary=paper.get("abstract"),
            tags=[paper.get("source")] if paper.get("source") else None,
            keywords=paper.get("title"),
        )
        conn.commit()
    finally:
        conn.close()
    return paper_id


def register_dataset(ds: dict) -> str:
    """登记数据集并同步统一索引（数据设计八.2）。"""
    dataset_id = ds.get("dataset_id") or uuid.uuid4().hex
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO dataset_registry(dataset_id, name, url, source, task_type, format,
                fields, labels, alignment, license, local_path, created_at, updated_at, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '1.0')
            """,
            (
                dataset_id,
                ds.get("name"),
                ds.get("url"),
                ds.get("source"),
                ds.get("task_type"),
                ds.get("format"),
                json.dumps(ds.get("fields"), ensure_ascii=False) if ds.get("fields") else None,
                json.dumps(ds.get("labels"), ensure_ascii=False) if ds.get("labels") else None,
                json.dumps(ds.get("alignment"), ensure_ascii=False) if ds.get("alignment") else None,
                ds.get("license"),
                ds.get("local_path"),
                _now(),
                _now(),
            ),
        )
        index_entry(
            conn,
            "dataset",
            dataset_id,
            title=ds.get("name"),
            summary=ds.get("source"),
            task_type=ds.get("task_type"),
            dataset_name=ds.get("name"),
            tags=[ds.get("source")] if ds.get("source") else None,
            keywords=ds.get("name"),
        )
        conn.commit()
    finally:
        conn.close()
    return dataset_id


def search(
    types: Optional[list[str]] = None,
    task_type: Optional[str] = None,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """统一检索（数据设计三.2）：类型/任务类型/模型/数据集过滤 + 关键词 FTS。"""
    sql = "SELECT * FROM unified_index WHERE 1=1"
    args: list = []

    if types:
        sql += f" AND data_type IN ({','.join('?' for _ in types)})"
        args.extend(types)
    if task_type:
        sql += " AND task_type = ?"
        args.append(task_type)
    if model:
        sql += " AND model_name = ?"
        args.append(model)
    if dataset:
        sql += " AND dataset_name = ?"
        args.append(dataset)
    if q:
        sql += " AND rowid IN (SELECT rowid FROM unified_index_fts WHERE unified_index_fts MATCH ?)"
        args.append(_fts_query(q))

    sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
    args.extend([limit, offset])

    conn = get_connection()
    try:
        rows = conn.execute(sql, args).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_item(data_type: str, ref_id: str) -> Optional[dict]:
    table_pk = _TABLE_PK.get(data_type)
    if table_pk is None:
        raise ValueError(f"unknown data_type: {data_type}")
    table, pk = table_pk
    conn = get_connection()
    try:
        row = conn.execute(f"SELECT * FROM {table} WHERE {pk} = ?", (ref_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def list_items(data_type: str, limit: int = 100, offset: int = 0) -> list[dict]:
    table_pk = _TABLE_PK.get(data_type)
    if table_pk is None:
        raise ValueError(f"unknown data_type: {data_type}")
    table, _ = table_pk
    conn = get_connection()
    try:
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def record_knowledge(k: dict) -> str:
    """写入蒸馏知识并同步统一索引（数据设计六.2）。"""
    knowledge_id = k.get("knowledge_id") or uuid.uuid4().hex
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO knowledge(knowledge_id, type, title, content, structured, sources,
                confidence, scope, status, created_at, updated_at, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '1.0')
            """,
            (
                knowledge_id,
                k.get("type"),
                k.get("title"),
                k.get("content"),
                json.dumps(k.get("structured"), ensure_ascii=False) if k.get("structured") else None,
                json.dumps(k.get("sources"), ensure_ascii=False) if k.get("sources") else None,
                k.get("confidence"),
                json.dumps(k.get("scope"), ensure_ascii=False) if k.get("scope") else None,
                k.get("status", "draft"),
                _now(),
                _now(),
            ),
        )
        index_entry(
            conn,
            "knowledge",
            knowledge_id,
            title=k.get("title"),
            summary=k.get("content"),
            tags=[k.get("type")] if k.get("type") else None,
            keywords=k.get("type"),
        )
        conn.commit()
    finally:
        conn.close()
    return knowledge_id


def confirm_knowledge(knowledge_id: str) -> bool:
    """蒸馏知识确认：draft → confirmed（数据设计六.3）。"""
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE knowledge SET status = 'confirmed', updated_at = ? WHERE knowledge_id = ? AND status = 'draft'",
            (_now(), knowledge_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_item(data_type: str, ref_id: str) -> bool:
    """删除条目，受引用约束（数据设计九.2）：被引用的数据禁止删除。"""
    table_pk = _TABLE_PK.get(data_type)
    if table_pk is None:
        raise ValueError(f"unknown data_type: {data_type}")
    table, pk = table_pk
    conn = get_connection()
    try:
        ref = conn.execute(
            "SELECT id FROM unified_index WHERE source_project_id = ? OR model_name = ? OR dataset_name = ? LIMIT 1",
            (ref_id, ref_id, ref_id),
        ).fetchone()
        if ref is not None:
            raise ReferenceError(f"{data_type}:{ref_id} 被其他条目引用，禁止删除")
        conn.execute("DELETE FROM unified_index WHERE data_type = ? AND ref_id = ?", (data_type, ref_id))
        cur = conn.execute(f"DELETE FROM {table} WHERE {pk} = ?", (ref_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def _match_scope(k: dict, task_type: Optional[str], model: Optional[str], dataset: Optional[str]) -> bool:
    scope = k.get("scope")
    if not scope:
        return True
    try:
        s = json.loads(scope)
    except (json.JSONDecodeError, TypeError):
        return True
    for key, val in (("task_type", task_type), ("model", model), ("dataset", dataset)):
        if val and s.get(key) and s[key] != val:
            return False
    return True


def bring_knowledge(
    task_type: Optional[str] = None,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
) -> dict:
    """任务前知识带入（数据设计三.3、模块详细设计八.2）：检索已确认蒸馏知识，按类型聚合为建议。"""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM knowledge WHERE status = 'confirmed' ORDER BY updated_at DESC"
        ).fetchall()
    finally:
        conn.close()

    param_advice, dependency_conflict, others = [], [], []
    for r in rows:
        k = dict(r)
        if not _match_scope(k, task_type, model, dataset):
            continue
        item = {"knowledge_id": k["knowledge_id"], "title": k["title"], "content": k["content"]}
        if k["structured"]:
            try:
                item["structured"] = json.loads(k["structured"])
            except json.JSONDecodeError:
                pass
        if k["type"] == "param_advice":
            param_advice.append(item)
        elif k["type"] == "dependency_conflict":
            dependency_conflict.append(item)
        else:
            others.append(item)

    return {"param_advice": param_advice, "dependency_conflict": dependency_conflict, "others": others}
