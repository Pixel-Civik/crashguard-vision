def test_allows_crashguard_dev_origin(client):
    response = client.options(
        "/sessions/session-id/report",
        headers={
            "origin": "https://crashguarddev.pixelcivik.com",
            "access-control-request-method": "GET",
            "access-control-request-headers": "x-vision-key",
        },
    )

    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "https://crashguarddev.pixelcivik.com"
    )
    assert "x-vision-key" in response.headers["access-control-allow-headers"].lower()


def test_rejects_unknown_origin(client):
    response = client.options(
        "/sessions/session-id/report",
        headers={
            "origin": "https://evil.example.com",
            "access-control-request-method": "GET",
            "access-control-request-headers": "x-vision-key",
        },
    )

    assert "access-control-allow-origin" not in response.headers
