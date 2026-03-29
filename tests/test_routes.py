"""Route integration tests."""


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
