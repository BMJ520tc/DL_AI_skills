"""检索与下载 API（模块详细设计 2.4）。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import download_service, knowledge_service

router = APIRouter(prefix="/api/search", tags=["search"])


class ExtractBody(BaseModel):
    """地址抽取请求体（向后兼容：仍可只传 paper_text）。

    - `paper_text`：论文正文（必填，保持既有契约）；
    - `paper_id`（可选）：论文 id，给了就顺带扫 `data/papers/<id>/` 下的补充材料
      （supplementary*、同目录 PDF/压缩包，正文本体 paper.pdf 不算）；
    - `supplementary_paths`（可选）：显式指定补充材料文件路径（PDF/压缩包/纯文本），与目录扫描合并去重；
    - `clone_repos`（可选，默认 true）：抽到仓库地址后是否立刻**完整克隆**到 `data/repos/`；
      设为 false 时只列出地址，不下载仓库（只想先看地址的场景）；
    - `full_clone`（可选，默认 true）：克隆口径。true=完整克隆（全历史 + 子模块 + Git LFS 尽力拉取）；
      false=浅克隆（--depth 1），仅供快速场景。
    """

    paper_text: str
    cwd: str | None = None
    paper_id: str | None = None
    supplementary_paths: list[str] | None = None
    clone_repos: bool = True
    full_clone: bool = True


class DownloadPaperBody(BaseModel):
    paper_id: str
    pdf_url: str
    title: str | None = None
    abstract: str | None = None
    source: str | None = None


class BatchPaperItem(BaseModel):
    """批量下载条目：pdf_url 可省，source=pubmed 时改抓 PMC OA 全文。"""

    paper_id: str
    pdf_url: str | None = None
    title: str | None = None
    abstract: str | None = None
    source: str | None = None


class DownloadPaperBatchBody(BaseModel):
    papers: list[BatchPaperItem]


def _split_csv(value: str | None) -> list[str] | None:
    """逗号分隔参数 → 去空白后的列表（用于 authors 多值）。"""
    if not value:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


@router.get("/papers")
def search_papers(
    q: str,
    source: str = "arxiv",
    max_results: int = 10,
    authors: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    fulltext: bool = False,
) -> list[dict]:
    """论文检索：新增 authors（逗号分隔）与 date_from/date_to（YYYY[-MM[-DD]]）过滤。

    bioRxiv 不支持服务端作者/时间过滤，会在结果条目里附 filter_note 如实说明。
    fulltext=true 时 PubMed 抓 PMC OA 全文、bioRxiv 走 Europe PMC / PMC OA 尽力抓全文
    （取不到只记 fulltext_status=abstract_only 与原因，不报错）。
    """
    author_list = _split_csv(authors)
    if source == "arxiv":
        return download_service.search_arxiv(
            q, max_results, authors=author_list, date_from=date_from, date_to=date_to
        )
    if source == "pubmed":
        return download_service.search_pubmed(
            q, max_results, authors=author_list, date_from=date_from, date_to=date_to, fulltext=fulltext
        )
    if source == "biorxiv":
        return download_service.search_biorxiv(
            q, max_results, authors=author_list, date_from=date_from, date_to=date_to, fulltext=fulltext
        )
    raise HTTPException(status_code=400, detail=f"不支持的 source: {source}（arxiv/pubmed/biorxiv）")


@router.post("/papers/download")
def download_paper(body: DownloadPaperBody) -> dict:
    paper_id = download_service.download_paper(
        body.paper_id, body.pdf_url, title=body.title, abstract=body.abstract, source=body.source
    )
    return {"paper_id": paper_id, "status": "downloaded"}


@router.post("/papers/download-batch")
def download_papers_batch(body: DownloadPaperBatchBody) -> dict:
    """批量下载论文（逐篇走既有单篇逻辑，单篇失败不影响其余）。

    部分失败不整体失败，逐篇返回成功/失败原因；PubMed 条目未给 pdf_url 时抓 PMC OA 全文，
    抓不到则记「仅摘要/无全文」（paper.status=abstract_only）。
    本接口为同步执行（逐篇串行下载）；批量很大时建议分批调用，避免一次请求长时间占用。
    """
    results: list[dict] = []
    for item in body.papers:
        try:
            paper_id = download_service.download_paper(
                item.paper_id, item.pdf_url,
                title=item.title, abstract=item.abstract, source=item.source,
            )
            record = knowledge_service.get_item("paper", paper_id) or {}
            results.append({"paper_id": paper_id, "status": record.get("status", "downloaded")})
        except Exception as e:  # noqa: BLE001 —— 单篇失败不中断批量
            results.append({"paper_id": item.paper_id, "status": "failed", "reason": str(e)})
    succeeded = sum(1 for r in results if r["status"] != "failed")
    return {
        "total": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "note": None if len(results) <= 20 else
                "批量较大（>20），本接口同步串行下载；更大批量建议分批调用，避免单次请求长时间占用。",
        "results": results,
    }


@router.post("/extract")
def extract_addresses(body: ExtractBody) -> dict:
    """提交地址抽取任务：从正文 + 补充材料抽仓库/数据集地址。

    抽到的数据集走下载登记；抽到的仓库地址逐个**完整克隆**到 `data/repos/<owner__repo>`
    （可用 clone_repos/full_clone 调整；单个仓库失败不中断其余，失败原因进任务进度与 run_record）。
    任务进度里可复核：repositories / repositories_cloned / repositories_failed /
    datasets_registered / datasets_failed / supplementary_used / supplementary_failed / warnings。
    """
    task_id = download_service.extract_addresses(
        body.paper_text,
        body.cwd,
        paper_id=body.paper_id,
        supplementary_paths=body.supplementary_paths,
        clone_repos=body.clone_repos,
        full_clone=body.full_clone,
    )
    return {"task_id": task_id, "status": "queued"}
