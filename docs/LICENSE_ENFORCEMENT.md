# License Enforcement - Technical Documentation

**How EDON v2 enforces licensing to protect intellectual property**

---

## Overview

EDON v2 implements multi-layer license enforcement to prevent unauthorized use and protect source code:

1. **Cloud Activation** - Initial activation via cloud server
2. **Periodic Validation** - Regular checks against activation server
3. **Expiration Enforcement** - 30-day evaluation period
4. **Revocation Support** - Remote revocation capability
5. **Signature Verification** - HMAC signatures prevent tampering

---

## Enforcement Points

### 1. Server Startup

License is validated when the server starts:

```python
# In app/main.py
if EDON_MODE == "v2":
    validate_license(force_online=True)
```

**Failure:** Server will not start (unless `EDON_SKIP_LICENSE_CHECK=true` for development)

### 2. Health Endpoint

`GET /health` includes license status and validates periodically:

```python
# License validation (periodic check for v2)
if mode == "v2" and LICENSING_AVAILABLE:
    validate_license(force_online=False)  # Use cached validation
```

**Failure:** Health endpoint returns 403 if license invalid

### 3. API Endpoints

All v2 endpoints validate license before processing:

- `POST /v2/oem/cav/batch` - Validates before processing
- `WS /v2/stream/cav` - Validates on connection
- gRPC endpoints - Validates before processing

**Failure:** Returns 403 (HTTP) or PERMISSION_DENIED (gRPC)

### 4. Periodic Validation

License is re-validated every hour (configurable):

```python
VALIDATION_INTERVAL = 3600  # 1 hour
```

**Offline Grace Period:** 24 hours allowed offline before enforcement

---

## License File Structure

License is stored in `~/.edon/license.json`:

```json
{
  "type": "evaluation",
  "version": "2.0.0",
  "key_version": "v1",
  "activated_at": 1234567890.0,
  "expires_at": 1237159890.0,
  "activation_id": "abc123",
  "org_id": "org-12345",
  "project_id": "project-atlas-v2",
  "plan": "evaluation",
  "hostname": "example-host",
  "revoked": false,
  "signature": "hmac-sha256-signature"
}
```

**Fields:**
- `key_version`: HMAC key version (for key rotation)
- `org_id`: Organization/company identifier
- `project_id`: Project/deployment identifier
- `plan`: License plan (evaluation, production, enterprise)
- `signature`: HMAC-SHA256 prevents tampering

---

## Activation Process

### Cloud Activation (Production)

1. Server contacts `https://activation.edon.ai/v1/activate`
2. Server receives activation ID and expiration
3. License file created with signature
4. License validated periodically

### Offline Activation (Development Only)

Enabled only if `EDON_ALLOW_OFFLINE_ACTIVATION=true`:

- Creates local license file
- 30-day expiration from activation
- No cloud validation (development only)

**Warning:** Offline activation is for development only and should be disabled in production builds.

---

## Validation Process

### Online Validation

1. Check license file exists and is valid
2. Verify HMAC signature
3. Check expiration date
4. Contact activation server
5. Server confirms license status
6. Update last validation time

### Offline Grace Period

If activation server is unreachable:

- Allow 24 hours offline operation
- After 24 hours, require online validation
- Prevents network issues from blocking legitimate use

---

## Clock Skew Handling

**Problem:** System clocks can be wrong, causing false expiration or infinite evaluation.

**Solution:**
- **Tolerance:** Allow up to 6 hours clock skew
- **Server Time:** Use server timestamp as source of truth
- **Compensation:** Automatically adjust for detected skew
- **Grace Period:** Expiration within skew tolerance is allowed (with warning)

**Implementation:**
- Server returns `server_time` in activation/validation responses
- Client calculates offset and applies compensation
- Expiration checks use adjusted time
- Large skew (>6 hours) is logged but not automatically corrected

**Benefits:**
- Prevents false expiration from clock drift
- Prevents infinite evaluation from clock set in past
- Handles timezone and NTP sync issues gracefully

---

## Key Rotation

**Problem:** Single HMAC secret is a security risk - if leaked, all licenses are compromised.

**Solution:**
- **Key Versions:** Support multiple key versions (v1, v2, etc.)
- **License Field:** Each license includes `key_version` field
- **Backward Compatibility:** Old key versions remain valid for transition period
- **Rotation Policy:** Document when old versions are deprecated

**Implementation:**
```python
LICENSE_SECRETS = {
    "v1": "secret-key-v1",
    "v2": "secret-key-v2",
}
CURRENT_KEY_VERSION = "v1"  # New licenses use this
MIN_SUPPORTED_KEY_VERSION = "v1"  # Oldest still accepted
```

**Rotation Process:**
1. Generate new key (v2)
2. Update `CURRENT_KEY_VERSION = "v2"`
3. New licenses use v2, old licenses still work with v1
4. After transition period, set `MIN_SUPPORTED_KEY_VERSION = "v2"`
5. v1 licenses become invalid (customers must re-activate)

**Benefits:**
- Security: Leaked key doesn't compromise all licenses
- Flexibility: Can rotate keys without breaking existing deployments
- Gradual migration: Old and new licenses coexist during transition

---

## Per-OEM + Per-Project Tracking

**Problem:** Need to track which organizations and projects are using EDON.

**Solution:**
- **org_id:** Organization/company identifier
- **project_id:** Project/deployment identifier
- **plan:** License plan (evaluation, production, enterprise)
- **activation_id:** Unique activation identifier

**License Structure:**
```json
{
  "org_id": "org-12345",
  "project_id": "project-atlas-v2",
  "plan": "evaluation",
  "activation_id": "act-abc123"
}
```

**Use Cases:**
- **Billing:** Track usage per organization/project
- **CRM:** Map activations to customer accounts
- **Analytics:** "Show all evals expiring this month"
- **Support:** "Which projects are using EDON?"
- **Compliance:** Audit trail of deployments

**Benefits:**
- Revenue tracking and billing
- Customer relationship management
- Usage analytics and reporting
- Compliance and audit trails

---

## Aggressive Logging

**Problem:** Need visibility into license operations for debugging and support.

**Solution:**
- **Structured Logging:** All license operations logged with structured data
- **Event Types:** Activation, validation, rejection events
- **Failure Reasons:** Detailed reasons for all failures
- **Audit Trail:** Complete history of license operations

**Logged Events:**

1. **Activation Attempts:**
   ```python
   {
     "event": "license_activation",
     "success": true/false,
     "reason": "...",
     "activation_id": "...",
     "org_id": "...",
     "project_id": "..."
   }
   ```

2. **Validation Results:**
   ```python
   {
     "event": "license_validation",
     "success": true/false,
     "reason": "...",
     "activation_id": "..."
   }
   ```

3. **Rejections:**
   ```python
   {
     "event": "license_rejection",
     "reason": "invalid_signature|expired|revoked|...",
     "activation_id": "..."
   }
   ```

**Use Cases:**
- **Support:** "Why did activation fail?" → Check logs
- **Analytics:** "How many active licenses?" → Query logs
- **Audits:** "Who is using expired licenses?" → Query logs
- **Debugging:** "Why is validation failing?" → Check logs

**Log Levels:**
- **INFO:** Successful operations (activation, validation)
- **WARNING:** Failures, rejections, clock skew
- **ERROR:** Critical errors, tampering attempts

---

## GDPR / Privacy Compliance

**Problem:** Enterprise customers require GDPR/privacy compliance.

**Solution:**
- **Minimal Data:** Only send anonymized license metadata
- **No Sensor Data:** Never send raw sensor data
- **No Personal Info:** Never send user IDs or personal information
- **Anonymization:** Hash hostnames and identifiers

**Data Transmitted:**

**Activation:**
```json
{
  "version": "2.0.0",
  "type": "evaluation",
  "timestamp": 1234567890.0,
  "hostname_hash": "abc123...",  // SHA256 hash, not actual hostname
  "org_id": "org-12345",  // Organization ID (not personal)
  "project_id": "project-atlas"  // Project ID (not personal)
}
```

**Validation:**
```json
{
  "activation_id": "act-abc123",
  "hostname_hash": "abc123...",  // Anonymized
  "org_id": "org-12345",
  "project_id": "project-atlas",
  "timestamp": 1234567890.0
}
```

**Explicitly NOT Sent:**
- ❌ Raw sensor data (EDA, BVP, ACC, etc.)
- ❌ Personal information
- ❌ User IDs or names
- ❌ Actual hostnames (only hashed)
- ❌ IP addresses (unless required for security)
- ❌ Any identifiable information about humans

**Compliance Statement:**
> "EDON license validation only transmits anonymized license metadata (organization ID, project ID, activation ID, hashed hostname). No raw sensor data, personal information, or user-identifiable data is transmitted. All data transmission is GDPR-compliant."

**Benefits:**
- Legal compliance (GDPR, CCPA)
- Enterprise customer trust
- Security team approval
- Reduced privacy liability

---

## Bypass Prevention

### 1. Signature Verification

License file is signed with HMAC-SHA256:

```python
signature = hmac.new(secret_key, license_data, hashlib.sha256).hexdigest()
```

**Tampering:** Invalid signature causes validation failure

**Key Rotation:** Multiple key versions supported for security

### 2. Server-Side Validation

Periodic checks against activation server:

- Cannot be bypassed by modifying local files
- Server can revoke licenses remotely
- Activation ID tracked server-side

### 3. Multiple Enforcement Points

License checked at:
- Server startup
- Health endpoint
- Every API request
- Periodic background validation

**Bypass Difficulty:** High - requires modifying multiple code paths

### 4. Source Code Protection

- License enforcement code is part of core application
- Removing enforcement breaks application
- Obfuscation/encryption can be added for production

---

## Development Bypass

For development/testing, bypass can be enabled:

```bash
export EDON_SKIP_LICENSE_CHECK=true
export EDON_ALLOW_OFFLINE_ACTIVATION=true
```

**Warning:** These should NEVER be enabled in production builds.

---

## Production Hardening

For production builds, additional protections:

1. **Remove bypass flags** - Compile-time removal
2. **Obfuscate code** - Make reverse engineering harder
3. **Encrypt license module** - Protect enforcement logic
4. **Hardware binding** - Bind to specific hardware IDs
5. **Air-gapped deployment** - Custom activation for enterprise

---

## Revocation

Licenses can be revoked remotely:

1. Server marks license as revoked
2. Next validation check fails
3. All endpoints return 403/PERMISSION_DENIED
4. License file updated with `revoked: true`

**Use Cases:**
- Evaluation period expired
- Terms violation
- Security breach
- Contract termination

---

## Error Messages

Common license errors:

- `"License not activated"` - No license file found
- `"License expired"` - Evaluation period ended
- `"License revoked"` - License revoked by server
- `"Invalid signature"` - License file tampered
- `"Server unreachable"` - Cannot contact activation server (after grace period)

---

## Configuration

Environment variables:

- `EDON_ACTIVATION_SERVER` - Activation server URL (default: `https://activation.edon.ai/v1/validate`)
- `EDON_LICENSE_SECRET_V1` - Secret key v1 for signing (should be server-side in production)
- `EDON_LICENSE_SECRET_V2` - Secret key v2 for signing (should be server-side in production)
- `EDON_SKIP_LICENSE_CHECK` - Bypass license check (development only)
- `EDON_ALLOW_OFFLINE_ACTIVATION` - Allow offline activation (development only)

**Key Rotation:**
- `CURRENT_KEY_VERSION` - Current key version for new licenses (code constant)
- `MIN_SUPPORTED_KEY_VERSION` - Oldest key version still accepted (code constant)

---

## Testing

To test license enforcement:

1. **Valid License:**
   ```bash
   # Activate license
   curl -X POST http://localhost:8001/license/activate
   ```

2. **Expired License:**
   ```bash
   # Manually expire (modify license file)
   # Or wait 30 days
   ```

3. **Revoked License:**
   ```bash
   # Server marks as revoked
   # Next validation fails
   ```

4. **Invalid Signature:**
   ```bash
   # Modify license file
   # Signature validation fails
   ```

---

## Security Considerations

1. **Secret Key:** Should be server-side only (not in client code)
2. **HTTPS:** Activation server should use HTTPS
3. **Rate Limiting:** Prevent activation abuse
4. **Audit Logging:** Track license activations and validations
5. **Hardware Binding:** For enterprise, bind to hardware IDs

---

## Future Enhancements

- Hardware fingerprinting
- Encrypted license files
- Code obfuscation
- Anti-debugging measures
- Tamper detection
- Air-gapped activation

---

**Last Updated:** 2025  
**Version:** 2.0.0

