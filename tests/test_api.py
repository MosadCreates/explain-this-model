import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from src.config import reload_config, set_config_path
from src.database import init_db


@pytest.fixture(autouse=True)
def _patch_db_and_config(monkeypatch):
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    from src import database
    database._engine = None
    database._SessionLocal = None
    init_db()

    from src import main
    main.get_config()
    reload_config()

    yield

    if database._engine is not None:
        database._engine.dispose()
    database._engine = None
    database._SessionLocal = None
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def client():
    from src.main import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_root_returns_service_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "Explain This Model"


class TestAnalyzeEndpoint:
    def test_analyze_accepts_valid_request(self, client, monkeypatch):
        def mock_delay(**kwargs):
            class MockAsyncResult:
                id = "mock-task-id"
            return MockAsyncResult()

        monkeypatch.setattr("src.api.router.run_analysis_task.delay", mock_delay)

        resp = client.post("/api/analyze", json={
            "model_name": "gpt2",
            "prompt": "Explain this model",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_analyze_rejects_empty_model_name(self, client):
        resp = client.post("/api/analyze", json={
            "model_name": "",
            "prompt": "test",
        })
        assert resp.status_code == 422

    def test_analyze_rejects_empty_prompt(self, client):
        resp = client.post("/api/analyze", json={
            "model_name": "gpt2",
            "prompt": "",
        })
        assert resp.status_code == 422

    def test_analyze_rejects_long_prompt(self, client):
        resp = client.post("/api/analyze", json={
            "model_name": "gpt2",
            "prompt": "x" * 5000,
        })
        assert resp.status_code == 422

    def test_analyze_called_without_auth_returns_201(self, client, monkeypatch):
        def mock_delay(**kwargs):
            class MockAsyncResult:
                id = "mock-task-id"
            return MockAsyncResult()

        monkeypatch.setattr("src.api.router.run_analysis_task.delay", mock_delay)

        resp = client.post("/api/analyze", json={
            "model_name": "distilbert-base-uncased",
            "prompt": "Test prompt",
        })
        assert resp.status_code == 201


class TestJobStatusEndpoint:
    def test_get_nonexistent_job_returns_404(self, client):
        resp = client.get("/api/jobs/nonexistent-id")
        assert resp.status_code == 404

    def test_get_pending_job_status(self, client, monkeypatch):
        def mock_delay(**kwargs):
            class MockAsyncResult:
                id = "mock-task-id"
            return MockAsyncResult()

        monkeypatch.setattr("src.api.router.run_analysis_task.delay", mock_delay)

        create_resp = client.post("/api/analyze", json={
            "model_name": "gpt2",
            "prompt": "test",
        })
        job_id = create_resp.json()["job_id"]

        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] == "pending"
        assert data["model_name"] == "gpt2"


class TestResultsEndpoint:
    def test_get_results_for_nonexistent_job(self, client):
        resp = client.get("/api/results/nonexistent")
        assert resp.status_code == 404

    def test_get_results_pending_job(self, client, monkeypatch):
        def mock_delay(**kwargs):
            class MockAsyncResult:
                id = "mock-task-id"
            return MockAsyncResult()

        monkeypatch.setattr("src.api.router.run_analysis_task.delay", mock_delay)

        create_resp = client.post("/api/analyze", json={
            "model_name": "gpt2",
            "prompt": "test",
        })
        job_id = create_resp.json()["job_id"]

        resp = client.get(f"/api/results/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"

    def test_get_completed_results(self, client, monkeypatch):
        from src.database import store_result

        def mock_delay(**kwargs):
            class MockAsyncResult:
                id = "mock-task-id"
            return MockAsyncResult()

        monkeypatch.setattr("src.api.router.run_analysis_task.delay", mock_delay)

        create_resp = client.post("/api/analyze", json={
            "model_name": "gpt2",
            "prompt": "test",
        })
        job_id = create_resp.json()["job_id"]

        store_result(job_id, {
            "model_name": "gpt2",
            "prompt": "test",
            "tokens": ["test"],
            "neuron_count": 3,
        })

        resp = client.get(f"/api/results/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["result"] is not None
        assert data["result"]["model_name"] == "gpt2"


class TestModelSearchEndpoint:
    def test_search_empty_query(self, client):
        resp = client.get("/api/models/search?query=")
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_search_short_query(self, client):
        resp = client.get("/api/models/search?query=a")
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_search_returns_list(self, client):
        resp = client.get("/api/models/search?query=gpt2&limit=3")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data


class TestConfigEndpoint:
    def test_config_endpoint_returns_config(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "analysis" in data
        assert "explanation" in data
        assert "model" in data


class TestModelValidateEndpoint:
    def test_validate_existing_model(self, client):
        resp = client.get("/api/models/validate?model_name=gpt2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert "parameter_count" in data
        assert "architecture" in data

    def test_validate_nonexistent_model(self, client):
        resp = client.get("/api/models/validate?model_name=this-model-does-not-exist-12345")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False


class TestRateLimiting:
    def test_rate_limit_after_max_jobs(self, client, monkeypatch):
        def mock_delay(**kwargs):
            class MockAsyncResult:
                id = "mock-task-id"
            return MockAsyncResult()

        monkeypatch.setattr("src.api.router.run_analysis_task.delay", mock_delay)
        from src.api.router import _rate_limiter
        original_window = _rate_limiter.window_seconds

        _rate_limiter.window_seconds = 3600
        _rate_limiter._requests.clear()

        max_req = _rate_limiter.max_requests
        for _ in range(max_req):
            resp = client.post("/api/analyze", json={
                "model_name": "gpt2",
                "prompt": "test",
            })
            assert resp.status_code == 201

        resp = client.post("/api/analyze", json={
            "model_name": "gpt2",
            "prompt": "test",
        })
        assert resp.status_code == 429

        _rate_limiter.window_seconds = original_window
        _rate_limiter._requests.clear()
