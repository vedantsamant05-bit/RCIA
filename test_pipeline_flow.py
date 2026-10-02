import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "status" in r.json()

def test_samples():
    r = client.get("/api/samples")
    assert r.status_code == 200
    samples = r.json()
    assert len(samples) > 0
    assert "regulatory_text" in samples[0]
    assert "text" in samples[0]

def test_pipeline_and_results():
    r_samples = client.get("/api/samples")
    samples = r_samples.json()
    sample_payload = {
        "title": "Master Direction on Periodic KYC Updation",
        "source": "RBI - Test Circular",
        "regulatory_text": samples[2]["regulatory_text"]
    }
    r = client.post("/api/pipeline", json=sample_payload)
    assert r.status_code == 200
    pipeline_res = r.json()
    assert "regulation_id" in pipeline_res
    reg_id = pipeline_res["regulation_id"]

    r_results = client.get(f"/api/regulations/{reg_id}/results")
    assert r_results.status_code == 200
    results_data = r_results.json()
    assert "regulation" in results_data
    assert "items" in results_data
    assert len(results_data["items"]) > 0

def test_review_and_audit():
    r_queue = client.get("/api/review")
    assert r_queue.status_code == 200
    assert isinstance(r_queue.json(), list)

    r_audit = client.get("/api/audit")
    assert r_audit.status_code == 200
    assert isinstance(r_audit.json(), list)

def test_pipeline_method_not_allowed():
    r_get = client.get("/api/pipeline")
    assert r_get.status_code == 405
    assert "detail" in r_get.json()

def test_openapi_schema():
    r_openapi = client.get("/openapi.json")
    assert r_openapi.status_code == 200
    paths = r_openapi.json()["paths"]
    assert "/api/pipeline" in paths
    assert "post" in paths["/api/pipeline"]

if __name__ == "__main__":
    test_health()
    test_samples()
    test_pipeline_and_results()
    test_review_and_audit()
    test_pipeline_method_not_allowed()
    test_openapi_schema()
    print("All backend tests passed!")
