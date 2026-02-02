# EDON SaaS Model

## Overview

EDON operates as a Software-as-a-Service (SaaS) platform, providing robot control intelligence through a unified API and web console. This document describes the complete user journey from signup to API access.

---

## User Journey

### Step 1: Account Creation

**What the user does:**

1. **Creates an EDON account** on your site (`https://edon.ai/signup`)
   - Provides email address
   - Optionally provides company name and use case

2. **Pays via Stripe**
   - Selects subscription plan (Pro, Pro+, Ultra)
   - Completes payment through Stripe Checkout
   - Receives email confirmation

3. **You provision them a tenant**
   - System generates unique `tenant_id` (e.g., `tenant_a1b2c3d4e5f6`)
   - Creates Stripe customer record
   - Sets initial subscription status to `trial` or `active`

### Step 2: Access Credentials

**They receive:**

1. **Console URL (UI):** `https://console.edon.ai`
   - Web-based dashboard for monitoring and configuration
   - Accessible via browser (no installation required)

2. **Gateway base URL (API):** `https://api.edon.ai/<tenant>`
   - RESTful API endpoint for programmatic access
   - Tenant-scoped: all requests are automatically scoped to their tenant

3. **API token (scoped to their tenant)**
   - Format: `edon_<random_hex_string>`
   - Generated automatically during tenant creation
   - Stored securely (hashed) in database
   - Can be regenerated/rotated via console

**Example:**
```
Tenant ID: tenant_a1b2c3d4e5f6
Console: https://console.edon.ai
API: https://api.edon.ai/tenant_a1b2c3d4e5f6
Token: edon_7f8e9d0c1b2a3f4e5d6c7b8a9f0e1d2c
```

---

## UI Access Enforcement

### Authentication Flow

**UI requires login:**

1. **Login Methods:**
   - **Email/Password:** Traditional username/password authentication
   - **Magic Link:** Passwordless email-based login (one-time link)
   - **Google OAuth:** Single sign-on via Google account

2. **Session Management:**
   - User logs in → backend creates session cookie
   - Session contains: `user_id`, `tenant_id`, `email`, `role`
   - Session expires after inactivity (configurable, default: 24 hours)

3. **API Token Exchange:**
   - UI calls gateway using user session
   - Backend exchanges session for short-lived API token (JWT)
   - Token valid for 1 hour, auto-refreshed
   - Alternative: Server-side proxy (UI → backend → gateway, no token exposure)

### Subscription Status Enforcement

**If subscription is inactive:**

1. **UI Behavior:**
   - Shows "Billing required" banner at top of console
   - Blocks all API calls (shows error message)
   - Displays "Upgrade" button linking to Stripe checkout
   - Read-only mode: can view historical data but cannot make changes

2. **API Behavior:**
   - All API requests return `402 Payment Required`
   - Response includes:
     ```json
     {
       "detail": "Subscription inactive. Status: canceled",
       "status": "canceled",
       "plan": "starter",
       "checkout_url": "https://checkout.stripe.com/..."
     }
     ```

3. **Subscription States:**
   - `trial`: Active trial period (14 days default)
   - `active`: Paid subscription active
   - `past_due`: Payment failed, grace period
   - `canceled`: Subscription canceled, access revoked
   - `inactive`: No active subscription

---

## Why This Model is Best

### 1. No "Download" Friction

**Traditional Model (Download):**
- User downloads software
- Installs dependencies
- Configures environment
- Troubleshoots installation issues
- Updates manually

**EDON SaaS Model:**
- ✅ Instant access via browser
- ✅ No installation required
- ✅ Automatic updates
- ✅ Works on any device (desktop, tablet, mobile)
- ✅ Zero setup time

### 2. Clean and Unavoidable Paywall

**Benefits:**
- ✅ **Clear value proposition:** Users see pricing upfront
- ✅ **No piracy:** Code never leaves your servers
- ✅ **Usage tracking:** Know exactly who uses what features
- ✅ **Easy upgrades:** One-click plan changes
- ✅ **Automatic enforcement:** System blocks access when payment fails

**Implementation:**
- Stripe Checkout for seamless payment
- Webhook integration for real-time subscription updates
- Middleware automatically checks subscription status on every request
- No way to bypass: authentication required for all endpoints

### 3. You Control Upgrades, Telemetry, and Support

**Upgrades:**
- ✅ **Roll out features gradually:** Feature flags per tenant
- ✅ **A/B testing:** Test new features with subset of users
- ✅ **Instant deployment:** Push updates without user action
- ✅ **Version control:** Support multiple API versions simultaneously

**Telemetry:**
- ✅ **Usage analytics:** Track API calls, endpoints used, error rates
- ✅ **Performance monitoring:** Response times, latency, throughput
- ✅ **Feature adoption:** Which features are most popular
- ✅ **Tenant health:** Identify struggling customers proactively

**Support:**
- ✅ **Centralized logging:** All requests logged with tenant context
- ✅ **Remote debugging:** Access logs without customer involvement
- ✅ **Proactive alerts:** Notify users of issues before they report
- ✅ **Self-service:** Console provides tools for common tasks

---

## Technical Implementation

## Plans (Current)

**Pro — $25/month**  
For solo builders, early teams, experimentation  
✅ AI agent governance (single project)  
✅ Policy-based allow/block decisions  
✅ Basic audit log (last 7 days)  
✅ Manual escalation to human  
✅ Single runtime / gateway  

**Pro+ — $60/month (Most Popular)**  
For teams running real agents in production  
✅ Everything in Pro  
✅ Multiple agents & integrations  
✅ Full decision audit trail (30–90 days)  
✅ 24/7 autonomous mode governance  
✅ Confidence scoring + decision reasoning  

**Ultra — Contact Sales**  
For companies where failure = real damage  
✅ Everything in Pro+  
✅ Unlimited agents & runtimes  
✅ Long-term audit retention (1+ year)  
✅ Custom policy packs (per use case)  
✅ Specialized governance for critical operations  

### Tenant Provisioning

**Signup Endpoint:** `POST /billing/signup`

```python
# Creates tenant
tenant_id = f"tenant_{uuid.uuid4().hex[:16]}"
db.create_tenant(tenant_id, email, stripe_customer_id)

# Generates API key
api_key = f"edon_{uuid.uuid4().hex}"
key_hash = hashlib.sha256(api_key.encode()).hexdigest()
db.create_api_key(tenant_id, key_hash, "Initial Key")
```

### Authentication Middleware

**Request Flow:**
```
1. Request arrives with X-EDON-TOKEN header
2. Middleware extracts token
3. Looks up token hash in database
4. Retrieves tenant info (status, plan)
5. Checks subscription status:
   - If inactive → 402 Payment Required
   - If active → Continue
6. Checks usage limits:
   - If exceeded → 429 Too Many Requests
   - If within limits → Continue
7. Adds tenant context to request state
8. Forwards to endpoint handler
```

### Subscription Status Check

**Middleware Code:**
```python
if tenant_status not in ["active", "trial"]:
    return JSONResponse(
        status_code=402,
        content={
            "detail": f"Subscription inactive. Status: {tenant_status}",
            "status": tenant_status,
            "plan": tenant_plan
        }
    )
```

### Usage Limits

**Per-Plan Limits:**
- **Starter:** 10,000 API calls/month, 500/day
- **Pro:** 100,000 API calls/month, 5,000/day
- **Enterprise:** Unlimited

**Enforcement:**
- Counted per tenant in database
- Reset monthly/daily based on plan
- Returns `429 Too Many Requests` when exceeded

---

## API Usage Example

### With API Token

```python
import requests

headers = {
    "X-EDON-TOKEN": "edon_7f8e9d0c1b2a3f4e5d6c7b8a9f0e1d2c",
    "Content-Type": "application/json"
}

# Make API call
response = requests.post(
    "https://api.edon.ai/tenant_a1b2c3d4e5f6/v1/control/tick",
    headers=headers,
    json={"window": [...], "robot_id": "robot_123"}
)

# If subscription inactive:
# Status: 402
# Body: {"detail": "Subscription inactive. Status: canceled"}
```

### With SDK

```python
from edon import EdonClient

client = EdonClient(
    api_key="edon_7f8e9d0c1b2a3f4e5d6c7b8a9f0e1d2c",
    base_url="https://api.edon.ai/tenant_a1b2c3d4e5f6"
)

# SDK automatically handles:
# - Token in headers
# - Error handling (402 → raises PaymentRequiredError)
# - Retry logic
# - Rate limiting

result = client.control_tick(window=[...], robot_id="robot_123")
```

---

## Security Considerations

### API Token Security

1. **Storage:**
   - Tokens stored as SHA-256 hashes (never plaintext)
   - Original token shown only once during creation
   - Tokens can be rotated without downtime

2. **Transmission:**
   - Always use HTTPS
   - Token sent in `X-EDON-TOKEN` header (not URL)
   - UI uses server-side proxy to avoid exposing tokens

3. **Scope:**
   - Tokens scoped to single tenant
   - Cannot access other tenants' data
   - Can be revoked instantly

### Session Security

1. **UI Sessions:**
   - HttpOnly cookies (not accessible to JavaScript)
   - Secure flag (HTTPS only)
   - SameSite protection (CSRF prevention)
   - Short expiration (24 hours default)

2. **Token Exchange:**
   - Short-lived JWTs (1 hour)
   - Auto-refresh before expiration
   - Revoked on logout

---

## Billing Integration

### Stripe Webhooks

**Events Handled:**
- `checkout.session.completed` → Activate subscription
- `invoice.paid` → Ensure subscription active
- `customer.subscription.updated` → Update plan/status
- `invoice.payment_failed` → Set status to `past_due`
- `customer.subscription.deleted` → Set status to `canceled`

**Webhook Endpoint:** `POST /billing/webhook`

### Subscription Lifecycle

```
1. User signs up → Status: "trial"
2. Payment completes → Status: "active"
3. Payment fails → Status: "past_due" (grace period)
4. Payment retried → Status: "active"
5. User cancels → Status: "canceled" (end of period)
6. Period ends → Status: "inactive" (access revoked)
```

---

## Multi-Tenancy Architecture

### Data Isolation

- **Database:** All tables include `tenant_id` column
- **Queries:** Automatically filtered by tenant context
- **API:** Tenant ID extracted from token, added to request state
- **Storage:** Tenant-specific data directories (if file storage)

### Resource Isolation

- **Compute:** Shared infrastructure, tenant quotas enforced
- **Rate Limiting:** Per-tenant limits (not global)
- **Logging:** All logs tagged with `tenant_id`
- **Metrics:** Aggregated per tenant for analytics

---

## Benefits Summary

### For Users

✅ **Zero installation:** Access from any browser  
✅ **Instant access:** Start using immediately after signup  
✅ **Automatic updates:** Always on latest version  
✅ **Scalable:** Handles growth automatically  
✅ **Secure:** Enterprise-grade security built-in  
✅ **Support:** Centralized support and documentation  

### For EDON

✅ **Predictable revenue:** Recurring subscriptions  
✅ **Usage insights:** Know exactly how customers use the platform  
✅ **Rapid iteration:** Deploy updates without customer coordination  
✅ **Security control:** Code never leaves your infrastructure  
✅ **Easy support:** Centralized logging and monitoring  
✅ **Scalable business:** Add customers without linear cost increase  

---

## Future Enhancements

### Planned Features

1. **Team Management:**
   - Multiple users per tenant
   - Role-based access control (RBAC)
   - Audit logs per user

2. **Usage Dashboard:**
   - Real-time usage metrics
   - Cost tracking
   - Usage alerts

3. **API Key Management:**
   - Multiple keys per tenant
   - Key rotation policies
   - Key expiration dates

4. **Enterprise Features:**
   - Single Sign-On (SSO) integration
   - Custom domains
   - Dedicated infrastructure
   - SLA guarantees

---

*Last Updated: 2025-01-27*
