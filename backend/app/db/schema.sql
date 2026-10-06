-- DL-AI-skills 知识库 schema
-- 依据《知识库与数据设计》四~八章；所有表带 schema_version（九.3）。
-- 结构化记录入 index.db；大文件（PDF/markdown/模块代码/运行产物）入文件系统，表中存路径引用。

-- 任务管理：调度状态（模块详细设计 2.1；结果/报错归 run_record，数据设计五.2）
CREATE TABLE IF NOT EXISTS task (
    task_id         TEXT PRIMARY KEY NOT NULL,
    task_type       TEXT NOT NULL,
    project_id      TEXT,
    params          TEXT,
    status          TEXT NOT NULL DEFAULT 'queued',   -- queued/running/success/failed/cancelled
    progress        TEXT,
    error           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    schema_version  TEXT NOT NULL DEFAULT '1.0'
);

-- 项目（数据设计八.1）
CREATE TABLE IF NOT EXISTS project (
    project_id        TEXT PRIMARY KEY NOT NULL,
    project_type      TEXT NOT NULL,                  -- original / structured
    source            TEXT,                           -- paper / repo / local / decompose / canvas
    parent_project_id TEXT,
    name              TEXT,
    status            TEXT NOT NULL DEFAULT 'loading',-- loading / env_ready / analyzed
    workspace_path    TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    schema_version    TEXT NOT NULL DEFAULT '1.0'
);

-- 论文（数据设计四.3）
CREATE TABLE IF NOT EXISTS paper (
    paper_id        TEXT PRIMARY KEY NOT NULL,
    title           TEXT,
    authors         TEXT,
    abstract        TEXT,
    source          TEXT,                             -- arxiv / pubmed / biorxiv
    url             TEXT,
    published_date  TEXT,
    license         TEXT,
    pdf_path        TEXT,
    markdown_path   TEXT,
    section_index   TEXT,                             -- JSON: 章节/表格/公式/图注位置索引
    status          TEXT NOT NULL DEFAULT 'downloaded',-- downloaded/parsed/extracted/reproduced/concluded
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    schema_version  TEXT NOT NULL DEFAULT '1.0'
);

-- 实验条目（数据设计四.3）
CREATE TABLE IF NOT EXISTS experiment_item (
    item_id               TEXT PRIMARY KEY NOT NULL,
    paper_id              TEXT NOT NULL,
    section_ref           TEXT,
    dataset_name          TEXT,
    split_method          TEXT,
    metric_name           TEXT,
    metric_value_reported TEXT,
    metric_unit           TEXT,
    hyperparams           TEXT,                        -- JSON
    baselines             TEXT,                        -- JSON
    status                TEXT NOT NULL DEFAULT 'extracted',
    schema_version        TEXT NOT NULL DEFAULT '1.0'
);

-- 复现对照（数据设计四.3）
CREATE TABLE IF NOT EXISTS reproduction_result (
    result_id           TEXT PRIMARY KEY NOT NULL,
    item_id             TEXT NOT NULL,
    run_id              TEXT,
    metric_value_actual TEXT,
    deviation           REAL,
    passed_threshold    INTEGER,
    verdict             TEXT,                          -- 一致/近似/不一致/无法复现
    evidence_path       TEXT,
    schema_version      TEXT NOT NULL DEFAULT '1.0'
);

-- 可信度结论（数据设计四.3）
CREATE TABLE IF NOT EXISTS credibility_conclusion (
    conclusion_id   TEXT PRIMARY KEY NOT NULL,
    paper_id        TEXT NOT NULL,
    overall_verdict TEXT,
    summary         TEXT,
    item_results    TEXT,                              -- JSON
    created_at      TEXT NOT NULL,
    schema_version  TEXT NOT NULL DEFAULT '1.0'
);

-- 论文↔项目 绑定留痕：同一篇论文可以用不同项目（各自的独立环境）复现，每次绑定都留档。
-- 复现任务创建时 upsert（按 paper_id + project_id 唯一），「复现板」据此列出用过的项目。
CREATE TABLE IF NOT EXISTS paper_project_binding (
    paper_id       TEXT NOT NULL,
    project_id     TEXT NOT NULL,
    created_at     TEXT NOT NULL,                  -- 首次绑定时间
    last_used_at   TEXT NOT NULL,                  -- 最近一次用它复现的时间
    uses           INTEGER NOT NULL DEFAULT 1,     -- 用它发起过几次复现
    last_task_id   TEXT,                           -- 最近一次复现任务 id（据此查那次的结果）
    schema_version TEXT NOT NULL DEFAULT '1.0',
    PRIMARY KEY (paper_id, project_id)
);

-- 运行记录（数据设计五.4）
CREATE TABLE IF NOT EXISTS run_record (
    run_id          TEXT PRIMARY KEY NOT NULL,
    project_id      TEXT,
    task_id         TEXT,
    run_type        TEXT NOT NULL,                     -- env_install/smoke_run/reproduce/baseline/eval/train
    environment     TEXT,                              -- JSON: 环境类型/语言/框架/CUDA/依赖清单
    params          TEXT,                              -- JSON
    command         TEXT,
    status          TEXT NOT NULL DEFAULT 'running',   -- success/failed/timeout
    metrics         TEXT,                              -- JSON
    error           TEXT,                              -- 参与 FTS 检索
    artifact_path   TEXT,
    log_path        TEXT,
    started_at      TEXT,
    finished_at     TEXT,
    duration_s      REAL,
    schema_version  TEXT NOT NULL DEFAULT '1.0'
);

-- 蒸馏知识（数据设计六.2）
CREATE TABLE IF NOT EXISTS knowledge (
    knowledge_id    TEXT PRIMARY KEY NOT NULL,
    type            TEXT NOT NULL,                     -- param_advice/dependency_conflict/reproduction_discrepancy/usage_guidance/fusion_insight
    title           TEXT,
    content         TEXT,
    structured      TEXT,                              -- JSON
    sources         TEXT,                              -- JSON: [paper_id/run_id/module_id]
    confidence      TEXT,                              -- high/medium/low
    scope           TEXT,                              -- JSON
    status          TEXT NOT NULL DEFAULT 'draft',     -- draft/confirmed/superseded
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    schema_version  TEXT NOT NULL DEFAULT '1.0'
);

-- 标准化模块（数据设计七；元信息与 module.json 对应）
CREATE TABLE IF NOT EXISTS module (
    module_id           TEXT NOT NULL,
    module_version      TEXT NOT NULL,
    name                TEXT,
    description         TEXT,
    source_project_id   TEXT,
    source_paper_id     TEXT,
    task_type           TEXT,
    input_spec          TEXT,                          -- JSON
    output_spec         TEXT,                          -- JSON
    params_schema       TEXT,                          -- JSON
    tags                TEXT,                          -- JSON
    verification        TEXT,                          -- JSON
    saved_module_compat TEXT,                          -- JSON
    path                TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    schema_version      TEXT NOT NULL DEFAULT '1.0',
    PRIMARY KEY (module_id, module_version)
);

-- 数据集登记（数据设计八.2）
CREATE TABLE IF NOT EXISTS dataset_registry (
    dataset_id      TEXT PRIMARY KEY NOT NULL,
    name            TEXT,
    url             TEXT,
    source          TEXT,                              -- zenodo/figshare/kaggle/自带
    task_type       TEXT,
    format          TEXT,                              -- CSV/Excel/FASTA/PDB/图片/压缩包
    fields          TEXT,                              -- JSON
    labels          TEXT,                              -- JSON
    alignment       TEXT,                              -- JSON: 字段映射/序列长度/标签归并
    license         TEXT,
    local_path      TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    schema_version  TEXT NOT NULL DEFAULT '1.0'
);

-- 统一索引（数据设计三.1）
CREATE TABLE IF NOT EXISTS unified_index (
    id                TEXT PRIMARY KEY NOT NULL,
    data_type         TEXT NOT NULL,                   -- paper/run/knowledge/module/dataset
    ref_id            TEXT NOT NULL,
    title             TEXT,
    summary           TEXT,
    source_project_id TEXT,
    task_type         TEXT,
    model_name        TEXT,
    dataset_name      TEXT,
    tags              TEXT,                            -- JSON
    keywords          TEXT,
    embedding         BLOB,                            -- O1 预留，一期不写入
    schema_version    TEXT NOT NULL DEFAULT '1.0',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

-- 统一索引全文检索（FTS5 external content，数据设计三.1）
CREATE VIRTUAL TABLE IF NOT EXISTS unified_index_fts USING fts5(
    title, summary, keywords, tags,
    content='unified_index',
    content_rowid='rowid'
);

-- FTS 同步触发器（保持与原表行对齐）
CREATE TRIGGER IF NOT EXISTS unified_index_ai AFTER INSERT ON unified_index BEGIN
    INSERT INTO unified_index_fts(rowid, title, summary, keywords, tags)
    VALUES (new.rowid, new.title, new.summary, new.keywords, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS unified_index_ad AFTER DELETE ON unified_index BEGIN
    INSERT INTO unified_index_fts(unified_index_fts, rowid, title, summary, keywords, tags)
    VALUES ('delete', old.rowid, old.title, old.summary, old.keywords, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS unified_index_au AFTER UPDATE ON unified_index BEGIN
    INSERT INTO unified_index_fts(unified_index_fts, rowid, title, summary, keywords, tags)
    VALUES ('delete', old.rowid, old.title, old.summary, old.keywords, old.tags);
    INSERT INTO unified_index_fts(rowid, title, summary, keywords, tags)
    VALUES (new.rowid, new.title, new.summary, new.keywords, new.tags);
END;
