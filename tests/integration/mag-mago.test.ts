/**
 * Integration tests for MAG orchestration flow
 */

import request from 'supertest';
import express from 'express';

// Note: These tests require MAG to be running with orchestration enabled
// Run with: npm run test:integration

const RUN = process.env.MAG_INTEGRATION_TESTS === 'true';
const describeIf = RUN ? describe : describe.skip;

describeIf('MAG Integration', () => {
  const MAG_URL = process.env.MAG_URL || 'http://localhost:8002';

  describe('End-to-End Flow', () => {
    it('should complete full flow: intent → authorization → command', async () => {
      // Step 1: Translate intent
      const intentResponse = await request(MAG_URL)
        .post('/mag/intent')
        .send({
          input: 'Shadow that vessel at 20km and avoid radar',
          source: 'text',
          context: {
            session_id: 'test_integration',
            operator_id: 'test_operator',
            domain: 'dod'
          },
          authorize: true
        });

      expect(intentResponse.status).toBe(200);
      expect(intentResponse.body.ok).toBe(true);
      expect(intentResponse.body.intent).toBeDefined();
      expect(intentResponse.body.authorization).toBeDefined();

      const intent = intentResponse.body.intent;
      const authorization = intentResponse.body.authorization;

      // Step 2: Verify authorization
      expect(authorization).toBeDefined();
      expect(authorization.decision).toBeDefined();
      expect(['allow', 'constrain', 'degrade']).toContain(authorization.decision);

      // Step 3: Translate to command (if authorized)
      if (authorization && authorization.decision !== 'deny') {
        const commandResponse = await request(MAG_URL)
          .post('/mag/adapters/uas_001/execute')
          .send({
            intent: intent
          });

        expect(commandResponse.status).toBe(200);
        expect(commandResponse.body.ok).toBe(true);
        expect(commandResponse.body.command).toBeDefined();
        expect(commandResponse.body.command.playbook_type).toBeDefined();
        expect(commandResponse.body.command.parameters).toBeDefined();
      }
    }, 10000); // 10 second timeout

    it('should handle search command flow', async () => {
      const intentResponse = await request(MAG_URL)
        .post('/mag/intent')
        .send({
          input: 'Search for target in 5km radius',
          source: 'text',
          context: {
            session_id: 'test_integration',
            domain: 'dod'
          },
          authorize: true
        });

      expect(intentResponse.status).toBe(200);
      expect(intentResponse.body.intent.structured_intent.action).toBe('search');
      expect(intentResponse.body.intent.structured_intent.parameters.radius_km).toBe(5);
    });

    it('should handle scan command flow', async () => {
      const intentResponse = await request(MAG_URL)
        .post('/mag/intent')
        .send({
          input: 'Scan the area for 10 minutes',
          source: 'text',
          context: {
            session_id: 'test_integration',
            domain: 'dod'
          },
          authorize: true
        });

      expect(intentResponse.status).toBe(200);
      expect(intentResponse.body.intent.structured_intent.action).toBe('scan');
      expect(intentResponse.body.intent.structured_intent.parameters.duration_seconds).toBe(600);
    });
  });

  describe('Error Handling', () => {
    it('should handle invalid intent gracefully', async () => {
      const response = await request(MAG_URL)
        .post('/mag/intent')
        .send({
          input: '', // Empty input
          source: 'text'
        });

      expect(response.status).toBe(400);
      expect(response.body.ok).toBe(false);
    });

    it('should handle missing adapter gracefully', async () => {
      const intent = {
        intent_id: 'test',
        timestamp: new Date().toISOString(),
        source: 'text',
        intent_type: 'task',
        natural_language: 'Test',
        structured_intent: {
          action: 'monitor'
        },
        context: {}
      };

      const response = await request(MAG_URL)
        .post('/mag/adapters/nonexistent_platform/execute')
        .send({
          intent: intent
        });

      expect(response.status).toBe(404);
      expect(response.body.ok).toBe(false);
    });
  });
});

