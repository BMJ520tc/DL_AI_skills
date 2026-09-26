"""检索与下载 API（模块详细设计 2.4）。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import download_service

router = APIRouter(prefix="/api/search", tags=["search"])


class ExtractBody(BaseModel):
    paper_text: str
    cwd: str | None = None


class DownloadPaperBody(BaseModel):
    paper_id: str
    pdf_url: str
    title: str | None = None
    abstract: str | None = None
    source: str | None = None


@router.get("/papers")
def search_papers(q: str, source: str = "arxiv", max_results: int = 10) -> list[dict]:
    if source == "arxiv":
        return download_service.search_arxiv(q, max_results)
    if source == "pubmed":
        return download_service.search_pubmed(q, max_results)
    if source == "biorxiv":
        return download_service.search_biorxiv(q, max_results)
    raise HTTPException(status_code=400, detail=f"不支持的 source: {source}（arxiv/pubmed/biorxiv）")


@router.post("/papers/download")
def download_paper(body: DownloadPaperBody) -> dict:
    paper_id = download_service.download_paper(
        body.paper_id, body.pdf_url, title=body.title, abstract=body.abstract, source=body.source
    )
    return {"paper_id": paper_id, "status": "downloaded"}


@router.post("/extract")
def extract_addresses(body: ExtractBody) -> dict:
    task_id = download_service.extract_addresses(body.paper_text, body.cwd)
    return {"task_id": task_id, "status": "queued"}
