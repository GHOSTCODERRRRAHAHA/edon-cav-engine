"""Tests for default intent updates via policy pack apply."""

import uuid

from fastapi.testclient import TestClient

from edon_gateway.config import config
from edon_gateway.main import app
from edon_gateway.persistence import get_db


def test_policy_pack_apply_updates_tenant_default_intent(monkeypatch):
    """Applying a pack should set tenant default intent."""
    token = "test-token"
    config._AUTH_ENABLED = True
    config._API_TOKEN = token

    tenant_id = f"tenant_dev_{uuid.uuid4().hex[:8]}"
    user_id = f"user_{uuid.uuid4().hex[:8]}"

    db = get_db()
    db.create_user(
        user_id=user_id,
        email=f"{user_id}@example.com",
        auth_provider="test",
        auth_subject=user_id
    )
    db.create_tenant(tenant_id=tenant_id, user_id=user_id)

    monkeypatch.setenv("EDON_DEV_TENANT_ID", tenant_id)
    monkeypatch.setenv("EDON_ENV", "development")

    client = TestClient(app)
    headers = {"Authorization": f"Bearer {token}"}

    response = client.post("/policy-packs/clawdbot_safe/apply", headers=headers)
    assert response.status_code == 200
    intent_id = response.json()["intent_id"]

    assert db.get_tenant_default_intent(tenant_id) == intent_id
