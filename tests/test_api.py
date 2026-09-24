import io


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["index"]["chunks_active"] > 0


def test_chat_endpoint_returns_citations(client):
    r = client.post("/api/chat", json={"message": "What is the SMA-2 classification threshold?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"]
    assert body["session_id"]
    assert body["retrieval"]["mode"] == "hybrid_rerank"
    assert "faithfulness" in body


def test_chat_keeps_session_history(client):
    first = client.post("/api/chat", json={"message": "What is the cooling off period for digital loans?"}).json()
    sid = first["session_id"]
    client.post("/api/chat", json={"message": "And the grievance redressal timeline?", "session_id": sid})
    history = client.get(f"/api/sessions/{sid}").json()
    assert len(history["messages"]) == 4  # 2 user + 2 assistant


def test_search_endpoint_modes(client):
    for mode in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
        r = client.post("/api/search", json={"query": "KYC record retention period", "mode": mode, "top_k": 3})
        assert r.status_code == 200, mode
        assert r.json()["results"], mode


def test_documents_listing_exposes_versions(client):
    docs = client.get("/api/documents").json()
    assert docs
    assert any(v["status"] in {"active", "deprecated"} for d in docs for v in d["versions"])


def test_upload_new_circular_updates_the_index(client):
    before = client.get("/api/health").json()["index"]["chunks_active"]
    content = b"""RBI/2027-28/15
DOR.TEST.REC.9/00.00.001/2027-28

Master Circular on Unit Test Deposits

01 April 2027

1. Scope
1.1 These directions apply to all scheduled commercial banks for the purposes of unit testing the ingestion pipeline of this system.

2. Threshold
2.1 The unit test deposit threshold shall be rupees seventeen thousand and three hundred, and banks shall report breaches within four working days of detection.
"""
    r = client.post(
        "/api/documents/upload",
        files={"file": ("unit-test-circular.txt", io.BytesIO(content), "text/plain")},
        data={"category": "Master Circular"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "indexed"
    assert body["document"]["doc_number"] == "RBI/2027-28/15"
    assert body["index"]["chunks_active"] > before

    # the brand-new content is immediately answerable
    answer = client.post("/api/chat", json={"message": "What is the unit test deposit threshold?"}).json()
    assert "seventeen thousand" in answer["answer"].lower() or any(
        "seventeen thousand" in c["text"].lower() for c in answer["citations"]
    )


def test_upload_rejects_unsupported_type(client):
    r = client.post(
        "/api/documents/upload",
        files={"file": ("x.exe", io.BytesIO(b"binary"), "application/octet-stream")},
    )
    assert r.status_code == 400


def test_add_document_via_pasted_text(client):
    text = (
        "RBI/2028-29/07\nMaster Direction on Pasted Text Handling\n\n01 June 2028\n\n"
        "1. Applicability\n1.1 This direction applies to all regulated entities that accept pasted circular text "
        "through the ingestion API of the regulation tracking system, and prescribes a reporting timeline of "
        "eleven working days for any deviation observed during processing.\n"
    ) * 2
    r = client.post("/api/documents/text", json={"text": text, "category": "Master Direction"})
    assert r.status_code == 200, r.text
    assert r.json()["document"]["doc_number"] == "RBI/2028-29/07"


def test_chunk_drilldown(client):
    results = client.post("/api/search", json={"query": "nomination facility deposit accounts", "top_k": 1}).json()
    chunk_id = results["results"][0]["chunk_id"]
    r = client.get(f"/api/chunks/{chunk_id}")
    assert r.status_code == 200
    assert r.json()["document"]["doc_number"]


def test_reindex_endpoint(client):
    r = client.post("/api/reindex")
    assert r.status_code == 200
    assert r.json()["index"]["chunks_total"] > 0
