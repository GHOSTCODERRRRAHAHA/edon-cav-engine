# Testing Guide

## Overview

This guide covers running tests for MAG (now unified with orchestration).

---

## Test Structure

```
edon-mag/
├── tests/
│   ├── policy-engine.test.ts
│   ├── risk-calculator.test.ts
│   └── uas-adapter.test.ts

tests/
├── integration/
│   └── mag-mago.test.ts
└── e2e/
    └── demo-scenario.test.ts
```

---

## Running Tests

### Unit Tests

**MAG:**
```bash
cd edon-mag
npm test
```

**Watch Mode:**
```bash
npm run test:watch
```

**Coverage:**
```bash
npm run test:coverage
```

### Integration Tests

**Prerequisites:** MAG server running with orchestration enabled.

```bash
# Terminal 1: Start MAG
cd edon-mag
npm run dev

# Terminal 2: Run integration tests
cd tests
MAG_INTEGRATION_TESTS=true MAG_URL=http://localhost:8002 npm test
```

### End-to-End Tests

**Prerequisites:** MAG server running with orchestration enabled.

```bash
cd tests
MAG_INTEGRATION_TESTS=true MAG_URL=http://localhost:8002 npm test
```

---

## Test Coverage

### Unit Tests

1. **Policy Engine** (`edon-mag/tests/policy-engine.test.ts`)
   - ✅ Rule evaluation
   - ✅ Priority ordering
   - ✅ Condition matching
   - ✅ Rule management

2. **Risk Calculator** (`edon-mag/tests/risk-calculator.test.ts`)
   - ✅ Risk assessment
   - ✅ Risk factor extraction
   - ✅ Mitigation suggestions

3. **UAS Adapter** (`edon-mag/tests/uas-adapter.test.ts`)
   - ✅ Command translation
   - ✅ Parameter extraction
   - ✅ Validation
   - ✅ Capabilities

### Integration Tests

1. **MAG Orchestration Flow** (`tests/integration/mag-mago.test.ts`)
   - ✅ End-to-end flow
   - ✅ Error handling
   - ✅ Multiple command types

### End-to-End Tests

1. **Demo Scenario** (`tests/e2e/demo-scenario.test.ts`)
   - ✅ Complete demo flow
   - ✅ Health checks
   - ✅ Intent → Authorization → Command

---

## Writing New Tests

### Unit Test Template

```typescript
import { Component } from '../src/component';

describe('Component', () => {
  let component: Component;

  beforeEach(() => {
    component = new Component();
  });

  describe('method', () => {
    it('should do something', () => {
      const result = component.method();
      expect(result).toBe(expected);
    });
  });
});
```

### Integration Test Template

```typescript
import request from 'supertest';

const URL = process.env.SERVICE_URL || 'http://localhost:8000';

describe('Integration Test', () => {
  it('should test flow', async () => {
    const response = await request(URL)
      .post('/endpoint')
      .send({ data: 'test' });

    expect(response.status).toBe(200);
    expect(response.body.ok).toBe(true);
  });
});
```

---

## Test Data

### Sample Intents

```typescript
const shadowIntent = {
  intent_id: 'test',
  timestamp: new Date().toISOString(),
  source: 'text',
  intent_type: 'task',
  natural_language: 'Shadow that vessel at 20km',
  structured_intent: {
    action: 'shadow',
    parameters: { distance_km: 20 }
  },
  context: {}
};
```

---

## Continuous Integration

Tests should run on:
- Pre-commit hooks
- Pull requests
- Main branch merges

---

## Coverage Goals

- **Unit Tests:** > 80% coverage
- **Integration Tests:** All critical flows
- **E2E Tests:** Demo scenario

---

## Troubleshooting

### Tests Fail with "Cannot find module"

Run `npm install` in the test directory.

### Integration Tests Fail

Ensure both MAG and MAGO servers are running.

### Timeout Errors

Increase timeout in test:
```typescript
it('should complete', async () => {
  // test code
}, 30000); // 30 second timeout
```

---

## Next Steps

- [ ] Add more unit tests for edge cases
- [ ] Add performance tests
- [ ] Add load tests
- [ ] Add security tests

