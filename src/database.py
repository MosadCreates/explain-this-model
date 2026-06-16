import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    status = Column(String(20), default="pending", index=True)
    model_name = Column(String(255), nullable=False)
    prompt = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    result_id = Column(String(36), ForeignKey("stored_results.id"), nullable=True)

    result = relationship("StoredResult", back_populates="job")


class StoredResult(Base):
    __tablename__ = "stored_results"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    json_data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    job = relationship("AnalysisJob", back_populates="result", uselist=False)


_engine = None
_SessionLocal = None


def get_db_path() -> str:
    return os.environ.get("DATABASE_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "explain_this_model.db"))


def get_db_url() -> str:
    return f"sqlite:///{get_db_path()}"


def get_engine():
    global _engine
    if _engine is None:
        db_path = get_db_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        _engine = create_engine(get_db_url(), connect_args={"check_same_thread": False})
        logger.info("Database engine created at %s", db_path)
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)
    return _SessionLocal


def init_db():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialised")


@contextmanager
def get_db():
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_job(model_name: str, prompt: str) -> AnalysisJob:
    job = AnalysisJob(
        model_name=model_name,
        prompt=prompt,
        status="pending",
    )
    with get_db() as db:
        db.add(job)
        db.flush()
        job_id = job.id
    return job_id


def get_job(job_id: str) -> Optional[AnalysisJob]:
    with get_db() as db:
        return db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()


def update_job_status(job_id: str, status: str, error_message: Optional[str] = None):
    with get_db() as db:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            job.status = status
            job.updated_at = datetime.now(timezone.utc)
            if status == "completed":
                job.completed_at = datetime.now(timezone.utc)
            if error_message:
                job.error_message = error_message


def store_result(job_id: str, result_data: dict) -> str:
    result_id = str(uuid4())
    with get_db() as db:
        stored = StoredResult(
            id=result_id,
            json_data=json.dumps(result_data, default=str),
        )
        db.add(stored)
        db.flush()

        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            job.result_id = result_id
            job.status = "completed"
            job.completed_at = datetime.now(timezone.utc)
            job.updated_at = datetime.now(timezone.utc)

    return result_id


def get_result(result_id: str) -> Optional[dict]:
    with get_db() as db:
        stored = db.query(StoredResult).filter(StoredResult.id == result_id).first()
        if stored:
            return json.loads(stored.json_data)
    return None


def delete_job(job_id: str):
    with get_db() as db:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            if job.result_id:
                db.query(StoredResult).filter(StoredResult.id == job.result_id).delete()
            db.delete(job)
