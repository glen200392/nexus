"""
PDF Reader Skill
Extract text, tables, and metadata from PDF files.
Primary: pdfplumber (best for tables)
Fallback: pypdf (pure Python, no binary deps)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from nexus.skills.registry import BaseSkill, SkillMeta

SKILL_META = SkillMeta(
    name="pdf_reader",
    description="Extract text, tables, and metadata from PDF files. "
                "Supports page-range selection, table extraction, and full-text search.",
    version="1.0.0",
    domains=["research", "analysis", "operations"],
    triggers=["pdf", "document", "extract", "parse", "read pdf", "PDF文件"],
    requires=["pdfplumber"],
    is_local=True,
)


class Skill(BaseSkill):
    meta = SKILL_META

    async def run(
        self,
        operation: str,          # extract_text | extract_tables | metadata | search | page_count
        file_path: str = "",
        pages: Optional[list[int]] = None,    # 0-indexed page numbers; None = all
        search_term: str = "",
        max_pages: int = 50,
        **kwargs,
    ) -> Any:
        if not file_path:
            return {"error": "file_path is required"}
        if not Path(file_path).exists():
            return {"error": f"File not found: {file_path}"}

        op = operation.lower()

        # Try pdfplumber first, fallback to pypdf
        try:
            return await self._run_pdfplumber(op, file_path, pages, search_term, max_pages)
        except ImportError:
            pass
        try:
            return await self._run_pypdf(op, file_path, pages, search_term, max_pages)
        except ImportError:
            return {"error": "No PDF library found. Install: pip install pdfplumber"}

    async def _run_pdfplumber(
        self, op: str, file_path: str, pages, search_term: str, max_pages: int
    ) -> Any:
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

            if op == "page_count" or op == "metadata":
                meta = pdf.metadata or {}
                return {
                    "page_count": total_pages,
                    "metadata": {
                        "title":    meta.get("Title", ""),
                        "author":   meta.get("Author", ""),
                        "creator":  meta.get("Creator", ""),
                        "subject":  meta.get("Subject", ""),
                        "keywords": meta.get("Keywords", ""),
                    },
                }

            page_indices = (
                [i for i in pages if 0 <= i < total_pages]
                if pages
                else list(range(min(total_pages, max_pages)))
            )

            if op == "extract_text":
                texts = {}
                for i in page_indices:
                    page = pdf.pages[i]
                    texts[i + 1] = page.extract_text() or ""
                full_text = "\n\n".join(
                    f"[Page {p}]\n{t}" for p, t in texts.items() if t.strip()
                )
                return {
                    "text":        full_text,
                    "pages_read":  len(page_indices),
                    "total_pages": total_pages,
                    "char_count":  len(full_text),
                }

            elif op == "extract_tables":
                all_tables = []
                for i in page_indices:
                    page   = pdf.pages[i]
                    tables = page.extract_tables()
                    for t in tables:
                        all_tables.append({"page": i + 1, "data": t})
                return {
                    "tables":      all_tables,
                    "table_count": len(all_tables),
                    "pages_read":  len(page_indices),
                }

            elif op == "search":
                if not search_term:
                    return {"error": "search_term required for search operation"}
                matches = []
                for i in page_indices:
                    page = pdf.pages[i]
                    text = page.extract_text() or ""
                    if search_term.lower() in text.lower():
                        # Find context around match
                        idx = text.lower().find(search_term.lower())
                        context = text[max(0, idx - 100): idx + 200]
                        matches.append({
                            "page":    i + 1,
                            "context": context,
                        })
                return {
                    "search_term": search_term,
                    "match_count": len(matches),
                    "matches":     matches,
                }

        return {"error": f"Unknown operation: {op}"}

    async def _run_pypdf(
        self, op: str, file_path: str, pages, search_term: str, max_pages: int
    ) -> Any:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        total  = len(reader.pages)

        if op in ("page_count", "metadata"):
            meta = reader.metadata or {}
            return {
                "page_count": total,
                "metadata": {
                    "title":  getattr(meta, "title",  "") or "",
                    "author": getattr(meta, "author", "") or "",
                },
            }

        page_indices = (
            [i for i in pages if 0 <= i < total]
            if pages
            else list(range(min(total, max_pages)))
        )
        texts = {i + 1: reader.pages[i].extract_text() or "" for i in page_indices}

        if op == "extract_text":
            full = "\n\n".join(f"[Page {p}]\n{t}" for p, t in texts.items() if t.strip())
            return {"text": full, "pages_read": len(page_indices), "total_pages": total}
        elif op == "search":
            matches = [
                {"page": p, "context": t[max(0, t.lower().find(search_term.lower()) - 80):
                                         t.lower().find(search_term.lower()) + 200]}
                for p, t in texts.items()
                if search_term.lower() in t.lower()
            ]
            return {"search_term": search_term, "match_count": len(matches), "matches": matches}

        return {"error": f"pypdf fallback: unsupported op '{op}'"}
