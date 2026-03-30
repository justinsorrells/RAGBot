"""Route integration tests."""

from pathlib import Path


def test_routes_support_upload_listing_chat_and_health(client) -> None:
    """The main API flow should work end to end for text uploads."""

    health_before = client.get("/health")
    assert health_before.status_code == 200
    assert health_before.json() == {"status": "ok", "documents_indexed": 0}

    upload = client.post(
        "/documents",
        files={
            "file": (
                "roadie.txt",
                b"Roadie is a crowdsourced same-day delivery platform for local logistics.",
                "text/plain",
            )
        },
    )
    assert upload.status_code == 201
    upload_payload = upload.json()
    assert upload_payload["filename"] == "roadie.txt"
    assert upload_payload["chunk_count"] >= 1

    listing = client.get("/documents")
    assert listing.status_code == 200
    listing_payload = listing.json()
    assert len(listing_payload) == 1
    assert listing_payload[0]["filename"] == "roadie.txt"

    chat = client.post("/chat", json={"question": "What does Roadie do?"})
    assert chat.status_code == 200
    chat_payload = chat.json()
    assert "Roadie" in chat_payload["answer"]
    assert chat_payload["sources"][0]["filename"] == "roadie.txt"
    assert "delivery platform" in chat_payload["sources"][0]["chunk_text"]

    health_after = client.get("/health")
    assert health_after.status_code == 200
    assert health_after.json()["documents_indexed"] == 1


def test_document_can_be_deleted_and_removed_from_retrieval(client) -> None:
    """Deleting a document should rebuild the index without its chunks."""

    upload = client.post(
        "/documents",
        files={
            "file": (
                "roadie.txt",
                b"Roadie coordinates same-day delivery for retailers and local logistics teams.",
                "text/plain",
            )
        },
    )
    document_id = upload.json()["document_id"]

    delete_response = client.delete(f"/documents/{document_id}")
    assert delete_response.status_code == 204

    listing = client.get("/documents")
    assert listing.status_code == 200
    assert listing.json() == []

    chat = client.post("/chat", json={"question": "What does Roadie coordinate?"})
    assert chat.status_code == 200
    assert chat.json()["sources"] == []


def test_document_can_be_reindexed_from_its_stored_source(client, test_settings) -> None:
    """Reindexing should refresh retrieval from the stored source file on disk."""

    upload = client.post(
        "/documents",
        files={"file": ("roadie.txt", b"Roadie delivers packages for merchants.", "text/plain")},
    )
    upload_payload = upload.json()
    document_id = upload_payload["document_id"]

    stored_path = Path(test_settings.document_store_path) / f"{document_id}.txt"
    stored_path.write_text(
        "Roadie provides same-day delivery software and driver coordination for local commerce.",
        encoding="utf-8",
    )

    reindex = client.post(f"/documents/{document_id}/reindex")
    assert reindex.status_code == 200
    assert reindex.json()["document_id"] == document_id

    chat = client.post("/chat", json={"question": "What does Roadie provide?"})
    assert chat.status_code == 200
    assert "driver coordination" in chat.json()["sources"][0]["chunk_text"]


def test_upload_rejects_unsupported_files(client) -> None:
    """Unsupported file uploads should return a 400 response."""

    response = client.post(
        "/documents",
        files={"file": ("notes.md", b"# markdown", "text/markdown")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_chat_returns_fallback_when_no_documents_are_indexed(client) -> None:
    """Chat should return a grounded fallback if no documents are available."""

    response = client.post("/chat", json={"question": "Anything indexed?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "I couldn't find relevant information in the indexed documents.",
        "sources": [],
    }
