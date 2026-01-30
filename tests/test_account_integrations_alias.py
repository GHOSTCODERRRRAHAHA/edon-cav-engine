"""Tests for /account/integrations alias behavior."""

import uuid

from fastapi.testclient import TestClient

from edon_gateway.config import config
from edon_gateway.main import app
from edon_gateway.persistence import get_db


def test_account_integrations_returns_connected_status(monkeypatch):
    """Ensure /account/integrations returns same status as integrations route."""
    token = "test-token"
    tenant_id = f"tenant_dev_{uuid.uuid4().hex[:8]}"

    monkeypatch.setattr(config, "_AUTH_ENABLED", True)
    monkeypatch.setattr(config, "_API_TOKEN", token)
    monkeypatch.setenv("EDON_ENV", "development")
    monkeypatch.setenv("EDON_DEV_TENANT_ID", tenant_id)

    db = get_db()
    db.save_credential(
        credential_id=f"clawdbot_gateway_{tenant_id}",
        tool_name="clawdbot",
        credential_type="gateway",
        credential_data={"base_url": "http://127.0.0.1:18789", "auth_mode": "token"},
        encrypted=False,
        tenant_id=tenant_id
    )

    client = TestClient(app)
    response = client.get("/account/integrations", headers={"X-EDON-TOKEN": token})
    alias_response = client.get("/integrations/account/integrations", headers={"X-EDON-TOKEN": token})

    assert response.status_code == 200
    assert alias_response.status_code == 200
    assert response.json() == alias_response.json()
    assert response.json()["clawdbot"]["connected"] is True
