import json
import os
import tempfile

import pytest

from src.database import (
    AnalysisJob,
    StoredResult,
    create_job,
    get_db,
    get_job,
    get_result,
    init_db,
    store_result,
    update_job_status,
)


@pytest.fixture(autouse=True)
def _patch_db_path(monkeypatch):
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    from src import database
    database._engine = None
    database._SessionLocal = None
    init_db()
    yield
    if database._engine is not None:
        database._engine.dispose()
    database._engine = None
    database._SessionLocal = None
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestCreateJob:
    def test_create_job_returns_id(self):
        job_id = create_job("gpt2", "Hello world")
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_created_job_has_pending_status(self):
        job_id = create_job("gpt2", "Test prompt")
        job = get_job(job_id)
        assert job.status == "pending"
        assert job.model_name == "gpt2"
        assert job.prompt == "Test prompt"


class TestGetJob:
    def test_get_nonexistent_job_returns_none(self):
        assert get_job("nonexistent-id") is None

    def test_get_job_returns_correct_model(self):
        job_id = create_job("distilbert", "Analyse this")
        job = get_job(job_id)
        assert job.model_name == "distilbert"


class TestUpdateJobStatus:
    def test_update_to_running(self):
        job_id = create_job("gpt2", "test")
        update_job_status(job_id, "running")
        job = get_job(job_id)
        assert job.status == "running"

    def test_update_to_failed_with_error(self):
        job_id = create_job("gpt2", "test")
        update_job_status(job_id, "failed", error_message="Something broke")
        job = get_job(job_id)
        assert job.status == "failed"
        assert job.error_message == "Something broke"

    def test_update_to_completed_sets_timestamp(self):
        job_id = create_job("gpt2", "test")
        update_job_status(job_id, "completed")
        job = get_job(job_id)
        assert job.status == "completed"
        assert job.completed_at is not None


class TestStoreAndGetResult:
    def test_store_and_retrieve_result(self):
        job_id = create_job("gpt2", "test prompt")
        result_data = {
            "model_name": "gpt2",
            "prompt": "test prompt",
            "tokens": ["test", "prompt"],
            "neuron_count": 5,
        }
        result_id = store_result(job_id, result_data)
        assert isinstance(result_id, str)

        retrieved = get_result(result_id)
        assert retrieved is not None
        assert retrieved["model_name"] == "gpt2"

    def test_store_result_links_to_job(self):
        job_id = create_job("gpt2", "test")
        store_result(job_id, {"key": "value"})
        job = get_job(job_id)
        assert job.status == "completed"
        assert job.result_id is not None

    def test_get_nonexistent_result(self):
        assert get_result("nonexistent") is None


class TestMultipleJobs:
    def test_multiple_jobs_isolated(self):
        id1 = create_job("gpt2", "prompt A")
        id2 = create_job("bert", "prompt B")
        assert id1 != id2

        job1 = get_job(id1)
        assert job1.model_name == "gpt2"

        job2 = get_job(id2)
        assert job2.model_name == "bert"
