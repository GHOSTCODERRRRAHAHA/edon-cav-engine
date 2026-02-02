/**
 * End-to-end test for demo scenario
 * "Shadow that vessel at 20km and avoid radar"
 */

import request from 'supertest';

const MAG_URL = process.env.MAG_URL || 'http://localhost:8002';
const RUN = process.env.MAG_INTEGRATION_TESTS === 'true';
const describeIf = RUN ? describe : describe.skip;

describeIf('Demo Scenario: Shadow Vessel', () => {
  const sessionId = `demo_test_${Date.now()}`;

  it('should complete demo scenario end-to-end', async () => {
    // Step 1: Health checks
    const magHealth = await request(MAG_URL).get('/health');
    expect(magHealth.status).toBe(200);

    // Step 2: Intent translation
    const intentResponse = await request(MAG_URL)
      .post('/mag/intent')
      .send({
        input: 'Shadow that vessel at 20km and avoid radar',
        source: 'text',
        context: {
          session_id: sessionId,
          operator_id: 'demo_operator',
          domain: 'dod'
        },
        authorize: true
      });

    expect(intentResponse.status).toBe(200);
    expect(intentResponse.body.ok).toBe(true);

    const intent = intentResponse.body.intent;
    const authorization = intentResponse.body.authorization;

    // Verify intent structure
    expect(intent.intent_id).toBeDefined();
    expect(intent.structured_intent.action).toBe('shadow');
    expect(intent.structured_intent.parameters.distance_km).toBe(20);
    expect(intent.structured_intent.parameters.avoid_radar).toBe(true);

    // Verify authorization
    expect(authorization).toBeDefined();
    expect(authorization.decision).toBeDefined();
    expect(authorization.decision_id).toBeDefined();
    expect(authorization.rationale).toBeDefined();

    // Step 3: Plan creation (MAG)
    const planResponse = await request(MAG_URL)
      .post('/mag/plan')
      .send({
        intent: intent
      });

    expect(planResponse.status).toBe(200);
    expect(planResponse.body.ok).toBe(true);
    expect(planResponse.body.plan_id).toBeDefined();
    expect(planResponse.body.proposed_steps).toBeDefined();

    // Step 4: Authorization (MAG)
    const magAuthorizeResponse = await request(MAG_URL)
      .post('/mag/authorize')
      .send({
        intent: intent,
        context: {
          session_id: sessionId,
          operator_id: 'demo_operator',
          domain: 'dod'
        }
      });

    expect(magAuthorizeResponse.status).toBe(200);
    expect(magAuthorizeResponse.body.ok).toBe(true);
    expect(magAuthorizeResponse.body.decision?.decision_id).toBeDefined();

    const magDecision = magAuthorizeResponse.body.decision;

    // Step 5: Command translation
    if (magDecision && magDecision.decision !== 'deny') {
      const commandResponse = await request(MAG_URL)
        .post('/mag/adapters/uas_001/execute')
        .send({
          intent: intent
        });

      expect(commandResponse.status).toBe(200);
      expect(commandResponse.body.ok).toBe(true);

      const command = commandResponse.body.command;

      // Verify command structure
      expect(command.command_id).toBeDefined();
      expect(command.platform_id).toBe('uas_001');
      expect(command.playbook_type).toBe('nav');
      expect(command.parameters.maintain_distance_km).toBe(20);
      expect(command.parameters.avoid_radar).toBe(true);
      expect(command.constraints).toBeDefined();
      expect(command.constraints.length).toBeGreaterThan(0);

      // Step 6: Explain decision (MAG)
      const explainResponse = await request(MAG_URL)
        .get(`/mag/decisions/${magDecision.decision_id}/explain`)
        .query({
          intent: JSON.stringify(intent),
          decision: JSON.stringify(magDecision)
        });

      if (explainResponse.status === 200) {
        expect(explainResponse.body.ok).toBe(true);
        expect(explainResponse.body.explanation).toBeDefined();
      }

      // Step 7: Verify ledger (if available)
      const ledgerResponse = await request(MAG_URL)
        .get(`/mag/ledger/events?session_id=${sessionId}`)
        .send();

      if (ledgerResponse.status === 200 && ledgerResponse.body.ok) {
        expect(ledgerResponse.body.events).toBeDefined();
        expect(ledgerResponse.body.events.length).toBeGreaterThan(0);
      }
    }
  }, 15000); // 15 second timeout for E2E test
});

