import json
import logging
import uuid
from typing import List, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import services


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["AI"])


# ============================================================
# Request Models
# ============================================================

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    chat_id: str = "default"
    collection_name: str = "rag_llm"


class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: List[ChatMessageIn] = Field(..., min_length=1)
    chat_id: str = "default"
    collection_name: str = "rag_llm"


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=300)
    count: int = Field(default=5, ge=1, le=20)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    qtype: Literal["mcq", "short-answer", "numerical"] = "mcq"
    collection_name: str = "rag_llm"


class SummarizeRequest(BaseModel):
    highlights: List[str] = Field(..., min_length=1, max_length=50)
    style: Literal["bullets", "paragraph"] = "bullets"


# ============================================================
# 1. ASK
# ============================================================

@router.post("/ask")
def ask(req: AskRequest):
    logger.info("ASK request: chat_id=%s", req.chat_id)

    try:
        answer, sources = services.answer_question(
            query=req.query,
            chat_id=req.chat_id,
            collection_name=req.collection_name,
        )

        if not answer:
            logger.warning("ASK returned empty answer")
            raise HTTPException(
                status_code=502,
                detail="Model returned an empty answer",
            )

        return {
            "success": True,
            "data": {
                "answer": answer,
                "sources": sources,
            },
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("ASK failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to answer question",
        )


# ============================================================
# 2. UPLOAD
# ============================================================

@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chapter_id: str = Form(...),
    collection_name: str = Form("rag_llm"),
):
    logger.info(
        "UPLOAD request: file=%s chapter_id=%s",
        file.filename,
        chapter_id,
    )

    if not (file.filename or "").lower().endswith(".pdf"):
        logger.warning("Rejected non-PDF upload: %s", file.filename)

        raise HTTPException(
            status_code=400,
            detail="Only .pdf files are accepted",
        )

    job_id = f"rag_job_{uuid.uuid4().hex[:8]}"

    try:
        saved_path = services.save_upload(file, job_id)

        services.create_job(
            job_id,
            file.filename,
            chapter_id,
            collection_name,
        )

        background_tasks.add_task(
            services.run_ingest,
            job_id,
            saved_path,
            chapter_id,
            collection_name,
        )

        logger.info("UPLOAD accepted: job_id=%s", job_id)

        return {
            "success": True,
            "message": "Upload accepted, indexing started",
            "data": {
                "jobId": job_id,
            },
        }

    except Exception:
        logger.exception("UPLOAD failed: job_id=%s", job_id)

        raise HTTPException(
            status_code=500,
            detail="Could not process upload",
        )

    finally:
        file.file.close()


@router.get("/upload/{job_id}")
def upload_status(job_id: str):
    logger.debug("Checking upload job: %s", job_id)

    job = services.get_job(job_id)

    if job is None:
        logger.warning("Unknown upload job: %s", job_id)

        raise HTTPException(
            status_code=404,
            detail=f"No such job: {job_id}",
        )

    return {
        "success": True,
        "data": job,
    }


# ============================================================
# 3. GENERATE
# ============================================================

@router.post("/generate")
def generate_questions(req: GenerateRequest):
    logger.info(
        "GENERATE request: topic=%s count=%s difficulty=%s",
        req.topic,
        req.count,
        req.difficulty,
    )

    try:
        questions = services.generate_questions(
            topic=req.topic,
            count=req.count,
            difficulty=req.difficulty,
            qtype=req.qtype,
            collection_name=req.collection_name,
        )

        if not questions:
            logger.warning(
                "No indexed content found for topic: %s",
                req.topic,
            )

            raise HTTPException(
                status_code=404,
                detail=f"No content indexed for topic: {req.topic}",
            )

        return {
            "success": True,
            "data": {
                "topic": req.topic,
                "questions": questions,
            },
        }

    except HTTPException:
        raise

    except json.JSONDecodeError:
        logger.exception("GENERATE returned invalid JSON")

        raise HTTPException(
            status_code=502,
            detail="Model returned malformed JSON",
        )

    except Exception:
        logger.exception("GENERATE failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to generate questions",
        )


# ============================================================
# 4. SUMMARIZE
# ============================================================

@router.post("/summarize")
def summarize(req: SummarizeRequest):
    logger.info(
        "SUMMARIZE request: %s highlights",
        len(req.highlights),
    )

    highlights = [h.strip() for h in req.highlights if h.strip()]

    if not highlights:
        logger.warning("SUMMARIZE received empty highlights")

        raise HTTPException(
            status_code=400,
            detail="All highlights were empty",
        )

    try:
        summary = services.summarize_highlights(
            highlights=highlights,
            style=req.style,
        )

        if not summary:
            logger.warning("SUMMARIZE returned empty result")

            raise HTTPException(
                status_code=502,
                detail="Model returned an empty summary",
            )

        return {
            "success": True,
            "data": {
                "summary": summary,
                "style": req.style,
            },
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("SUMMARIZE failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to summarise",
        )


# ============================================================
# 5. CHAT
# ============================================================

@router.post("/chat")
def chat(req: ChatRequest):
    logger.info(
        "CHAT request: chat_id=%s messages=%s",
        req.chat_id,
        len(req.messages),
    )

    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"),
        None,
    )

    if not last_user:
        logger.warning("CHAT request contains no user message")

        raise HTTPException(
            status_code=400,
            detail="No user message in the conversation",
        )

    try:
        token_stream = services.stream_chat(
            query=last_user,
            chat_id=req.chat_id,
            collection_name=req.collection_name,
            messages=[message.model_dump() for message in req.messages],
        )

    except Exception:
        logger.exception(
            "CHAT failed to start: chat_id=%s",
            req.chat_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to start chat",
        )

    def stream():
        try:
            for chunk in token_stream:
                yield chunk

        except Exception:
            logger.exception(
                "CHAT streaming failed: chat_id=%s",
                req.chat_id,
            )

            yield "data: " + json.dumps({
                "type": "error",
                "text": "Something went wrong. Please retry.",
            }) + "\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
    )
