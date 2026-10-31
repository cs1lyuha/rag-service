"""API-ul HTTP.

    GET    /health
    GET    /documents                    listă (cheie de utilizator)
    PUT    /documents/{name}             upsert Markdown (cheie de admin) → reindexare incrementală
    DELETE /documents/{name}             (cheie de admin)
    POST   /ask        {question, k}     răspuns + citări (cheie de utilizator)
    POST   /ask/stream {question, k}     SSE: event delta (text), apoi event done (citări)

Chei: RAG_API_KEYS (separate prin virgulă) pentru utilizatori, RAG_ADMIN_KEY pentru ingestie.
"""

from __future__ import annotations

import hmac
import json
import os
import time
from pathlib import Path
from typing import Iterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ragsvc.answer import Answerer, ExtractiveAnswerer, answer, clean_citations
from ragsvc.index import Chunk, HybridIndex


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=3, ge=1, le=8)


def citation_json(c: Chunk) -> dict:
    return {"id": c.id, "document": c.title, "section": c.section, "snippet": c.text[:240]}


def create_app(index: HybridIndex | None = None, answerer: Answerer | None = None, *,
               user_keys: set[str] | None = None, admin_key: str | None = None,
               kb_dir: str | Path | None = None) -> FastAPI:
    idx = index or HybridIndex()
    ans = answerer or ExtractiveAnswerer()
    users = user_keys if user_keys is not None else {k for k in os.environ.get("RAG_API_KEYS", "").split(",") if k}
    admin = admin_key if admin_key is not None else os.environ.get("RAG_ADMIN_KEY", "")
    folder = kb_dir if kb_dir is not None else os.environ.get("RAG_KB_DIR")
    if folder:
        for p in sorted(Path(folder).glob("*.md")):
            idx.upsert(p.stem, p.read_text(encoding="utf-8"))
    app = FastAPI(title="rag-service", version="0.1.0")

    def _check(key: str | None, allowed: set[str]) -> None:
        if not key or not any(hmac.compare_digest(key, k) for k in allowed):
            raise HTTPException(401, "cheie API invalidă")

    def user(x_api_key: str | None = Header(default=None)) -> None:
        _check(x_api_key, users | ({admin} if admin else set()))

    def admin_only(x_api_key: str | None = Header(default=None)) -> None:
        _check(x_api_key, {admin} if admin else set())

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "documents": len(idx.docs), "chunks": len(idx.chunks)}

    @app.get("/documents", dependencies=[Depends(user)])
    def documents() -> list[dict]:
        counts: dict[str, int] = {}
        for c in idx.chunks:
            counts[c.doc] = counts.get(c.doc, 0) + 1
        return [{"name": n, "sections": counts.get(n, 0), "sha256": h[:12]} for n, h in sorted(idx.docs.items())]

    @app.put("/documents/{name}", dependencies=[Depends(admin_only)])
    async def put_document(name: str, request: Request) -> dict:
        body = (await request.body()).decode("utf-8", errors="replace")
        if not body.strip() or len(body) > 500_000:
            raise HTTPException(422, "documentul e gol sau prea mare (max 500 KB)")
        try:
            status = idx.upsert(name, body)
        except ValueError as err:
            raise HTTPException(422, str(err)) from None
        return {"name": name, "status": status, "chunks": len(idx.chunks)}

    @app.delete("/documents/{name}", dependencies=[Depends(admin_only)])
    def delete_document(name: str) -> dict:
        if not idx.delete(name):
            raise HTTPException(404, "document inexistent")
        return {"name": name, "status": "deleted"}

    @app.post("/ask", dependencies=[Depends(user)])
    def ask(body: AskIn) -> dict:
        t0 = time.perf_counter()
        sources = [c for c, _, _ in idx.search(body.question, k=body.k)]
        result = answer(ans, body.question, sources)
        return {"answer": result.text, "abstained": result.abstained,
                "citations": [citation_json(c) for c in result.citations],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    @app.post("/ask/stream", dependencies=[Depends(user)])
    def ask_stream(body: AskIn) -> StreamingResponse:
        sources = [c for c, _, _ in idx.search(body.question, k=body.k)]

        def events() -> Iterator[str]:
            parts: list[str] = []
            try:
                for piece in ans.stream(body.question, sources):
                    parts.append(piece)
                    if "NU_STIU" not in piece:
                        yield f"event: delta\ndata: {json.dumps({'text': piece}, ensure_ascii=False)}\n\n"
            except Exception as err:  # providerul a căzut la mijloc: clientul află explicit
                yield f"event: error\ndata: {json.dumps({'error': type(err).__name__}, ensure_ascii=False)}\n\n"
                return
            final = clean_citations("".join(parts), sources)
            payload = {"answer": final.text, "abstained": final.abstained,
                       "citations": [citation_json(c) for c in final.citations]}
            yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
