# Auth Provider Migration Guide

## Overview

EDON uses an **auth provider abstraction layer** that makes it easy to migrate between authentication providers (Clerk → Supabase, etc.) without rewriting your application logic.

---

## Core Principles

### 1. Never Use Auth Provider IDs as Real User IDs

**✅ DO:**
- Generate internal UUIDs for `user_id`
- Store auth provider IDs as aliases (`auth_provider`, `auth_subject`)

**❌ DON'T:**
- Use `clerk_user_id` as your primary key
- Reference auth provider IDs in foreign keys

**Example:**
```python
# Users table
user_id = "550e8400-e29b-41d4-a716-446655440000"  # Internal UUID (never changes)
auth_provider = "clerk"
auth_subject = "user_2abc123def456"  # Clerk's user ID (can change)

# Later migration:
auth_provider = "supabase"
auth_subject = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"  # Supabase's user ID
# user_id stays the same!
```

### 2. Keep Authorization Inside EDON

**Auth Vendor Responsibilities:**
- Identity verification (who is this user?)
- Session management (is this token valid?)

**EDON Backend Responsibilities:**
- Permissions (what can this user do?)
- Policy enforcement (tenant boundaries)
- Subscription status (can they access this feature?)

**Why This Matters:**
- Switching auth = swapping token validation
- Your app logic stays unchanged
- Authorization rules remain in your control

### 3. Use a Single Session Claims Contract

**Standardized Contract:**
```python
class SessionClaims:
    user_id: str        # Internal UUID (never changes)
    tenant_id: str      # Tenant UUID
    email: str
    role: str           # 'user', 'admin', etc.
    plan: str           # 'starter', 'pro', 'enterprise'
    status: str         # 'active', 'trial', 'past_due', etc.
```

**All auth providers must yield this contract.**

---

## Database Schema

### Users Table (Auth Provider Agnostic)

```sql
CREATE TABLE users (
    id TEXT PRIMARY KEY,                    -- Internal UUID
    email TEXT NOT NULL UNIQUE,
    auth_provider TEXT NOT NULL,            -- 'clerk', 'supabase', etc.
    auth_subject TEXT NOT NULL,             -- Provider's user ID
    role TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(auth_provider, auth_subject)     -- One user per provider ID
);
```

### Tenants Table (References Users)

```sql
CREATE TABLE tenants (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,                  -- References users.id (internal UUID)
    status TEXT NOT NULL DEFAULT 'trial',
    plan TEXT NOT NULL DEFAULT 'free',
    stripe_customer_id TEXT UNIQUE,
    -- ... other fields
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

---

## Implementation

### Current: Clerk Integration

**1. User Signs Up via Clerk:**
```typescript
// Frontend: Clerk handles signup
<SignUp />
```

**2. Backend Links Clerk Account:**
```python
# POST /auth/signup
{
    "auth_provider": "clerk",
    "auth_subject": "user_2abc123def456",  # From Clerk
    "email": "user@example.com"
}

# Backend creates:
# - users record with internal UUID
# - tenants record linked to user
# - Returns session_token
```

**3. Token Validation:**
```python
# POST /auth/session
# Header: Authorization: Bearer <clerk_token>

# Backend validates Clerk token, returns SessionClaims
def validate_clerk_token(token: str) -> SessionClaims:
    # 1. Verify Clerk JWT signature
    # 2. Extract clerk_user_id
    # 3. Look up user by auth_provider='clerk', auth_subject=clerk_user_id
    # 4. Get tenant for user
    # 5. Return SessionClaims
```

### Future: Supabase Migration

**What Changes:**
1. **Token Validation Function:**
   ```python
   def validate_supabase_token(token: str) -> SessionClaims:
       # Replace Clerk JWT verification with Supabase JWT verification
       # Everything else stays the same
   ```

2. **Frontend UI:**
   - Replace Clerk components with Supabase components
   - Update signup/login flows

3. **Database:**
   - Update `auth_provider` from `'clerk'` to `'supabase'`
   - Update `auth_subject` to Supabase user IDs
   - **No changes to `user_id` or `tenant_id`**

**What Stays the Same:**
- ✅ All authorization logic
- ✅ Tenant boundaries
- ✅ Subscription checks
- ✅ API endpoints
- ✅ Database schema (except auth_provider/auth_subject values)

---

## Migration Steps (Clerk → Supabase)

### Step 1: Update Token Validation

**Before (Clerk):**
```python
def validate_clerk_token(token: str) -> SessionClaims:
    decoded = jwt.decode(token, CLERK_PUBLIC_KEY, algorithms=["RS256"])
    clerk_user_id = decoded["sub"]
    user = db.get_user_by_auth("clerk", clerk_user_id)
    # ... rest of logic
```

**After (Supabase):**
```python
def validate_supabase_token(token: str) -> SessionClaims:
    decoded = jwt.decode(token, SUPABASE_PUBLIC_KEY, algorithms=["HS256"])
    supabase_user_id = decoded["sub"]
    user = db.get_user_by_auth("supabase", supabase_user_id)
    # ... rest of logic (identical!)
```

### Step 2: Update User Records

```sql
-- Migrate existing users
UPDATE users 
SET auth_provider = 'supabase',
    auth_subject = '<supabase_user_id>'
WHERE auth_provider = 'clerk'
AND auth_subject = '<clerk_user_id>';

-- user_id stays the same!
```

### Step 3: Update Frontend

- Replace `<ClerkProvider>` with Supabase client
- Replace `<SignInButton>` with Supabase sign-in
- Update token handling

---

## Session Claims Contract

**All auth providers must implement this contract:**

```python
class SessionClaims(BaseModel):
    user_id: str        # Internal UUID (stable across migrations)
    tenant_id: str      # Tenant UUID
    email: str
    role: str           # 'user', 'admin'
    plan: str           # 'starter', 'pro', 'enterprise'
    status: str         # 'active', 'trial', 'past_due', 'canceled', 'inactive'
```

**Usage:**
```python
# Validate token and get claims
claims = validate_auth_token(token)  # Works with any provider

# Use claims for authorization
if claims.status != "active":
    raise HTTPException(402, "Subscription inactive")

# All your app logic uses claims, not provider-specific data
```

---

## Benefits of This Approach

### 1. Stable User IDs
- `user_id` never changes, even when switching auth providers
- Foreign keys remain valid
- No data migration needed

### 2. Easy Provider Switching
- Change token validation function
- Update `auth_provider` and `auth_subject` values
- Done!

### 3. Multi-Provider Support
- Users can link multiple auth providers
- `UNIQUE(auth_provider, auth_subject)` allows:
  - Same user with Clerk + Google OAuth
  - Same user with Supabase + GitHub

### 4. Authorization Stays in Your Control
- Auth provider = identity only
- EDON backend = permissions + policy
- No vendor lock-in for authorization logic

---

## Example: Clerk → Supabase Migration

**Before:**
```python
# Clerk token validation
claims = validate_clerk_token(token)
# Returns: SessionClaims(user_id="...", tenant_id="...", ...)
```

**After:**
```python
# Supabase token validation
claims = validate_supabase_token(token)
# Returns: SessionClaims(user_id="...", tenant_id="...", ...)
# Same contract, different implementation!
```

**Database:**
```sql
-- Before
SELECT * FROM users WHERE auth_provider = 'clerk' AND auth_subject = 'user_2abc123';

-- After (same user_id!)
SELECT * FROM users WHERE auth_provider = 'supabase' AND auth_subject = 'a1b2c3d4-e5f6-7890';
```

---

## API Endpoints

### `POST /auth/signup`
Create or link auth provider account.

**Request:**
```json
{
    "auth_provider": "clerk",
    "auth_subject": "user_2abc123def456",
    "email": "user@example.com"
}
```

**Response:**
```json
{
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "tenant_id": "tenant_a1b2c3d4e5f6",
    "session_token": "edon_session_..."
}
```

### `POST /auth/session`
Validate auth token and return session claims.

**Request:**
```
Authorization: Bearer <clerk_token>
```

**Response:**
```json
{
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "tenant_id": "tenant_a1b2c3d4e5f6",
    "email": "user@example.com",
    "role": "user",
    "plan": "starter",
    "status": "active"
}
```

---

## Summary

**Key Takeaways:**

1. ✅ **Internal UUIDs** for `user_id` (never use auth provider IDs)
2. ✅ **Auth provider as alias** (`auth_provider`, `auth_subject`)
3. ✅ **Standardized session claims** contract
4. ✅ **Authorization in EDON**, identity in auth provider
5. ✅ **Easy migration** = swap token validation function

**Migration Checklist:**

- [ ] Database schema updated (users table with auth_provider/auth_subject)
- [ ] Token validation function implemented
- [ ] Session claims contract standardized
- [ ] All endpoints use SessionClaims (not provider-specific data)
- [ ] Frontend uses auth provider components
- [ ] Backend authorization logic uses SessionClaims

---

*Last Updated: 2025-01-27*
