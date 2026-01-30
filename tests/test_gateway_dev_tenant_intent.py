"""Tests for dev tenant default intent flow."""

from fastapi.testclient import TestClient

from edon_gateway.config import config
from edon_gateway.main import app


def test_dev_token_sets_tenant_default_intent(monkeypatch):
    """EDON_API_TOKEN requests should use a stable dev tenant_id."""
    dev_tenant_id = "tenant_dev_test"
    monkeypatch.setenv("EDON_DEV_TENANT_ID", dev_tenant_id)

    client = TestClient(app)
    headers = {"Authorization": f"Bearer {config.API_TOKEN}"}

    apply_response = client.post("/policy-packs/clawdbot_safe/apply", headers=headers)
    assert apply_response.status_code == 200
    intent_id = apply_response.json()["intent_id"]

    integrations_response = client.get("/account/integrations", headers=headers)
    assert integrations_response.status_code == 200
    default_intent_id = integrations_response.json()["clawdbot"]["default_intent_id"]

    assert default_intent_id is not None
    assert default_intent_id == intent_id
