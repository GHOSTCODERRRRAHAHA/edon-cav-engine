# Auth Provider Abstraction Implementation

## ✅ What Was Implemented

### 1. Database Schema Updates

**New `users` Table:**
- `id` (TEXT PRIMARY KEY) - Internal UUID (never changes)
- `email` (TEXT UNIQUE)
- `auth_provider` (TEXT) - 'clerk', 'supabase', etc.
- `auth_subject` (TEXT) - Provider's user ID
- `role` (TEXT) - 'user', 'admin', etc.
- `created_at`, `updated_at`
- `UNIQUE(auth_provider, auth_subject)` constraint

**Updated `tenants` Table:**
- Now references `user_id` (FOREIGN KEY to `users.id`)
- Removed direct `email` field (now comes from `users` table via JOIN)

**Key Principle:**
- ✅ Internal UUIDs for `user_id` (stable across migrations)
- ✅ Auth provider IDs stored as aliases only
- ✅ Database stays stable when switching providers

### 2. Database Methods

**New User Methods:**
- `create_user(user_id, email, auth_provider, auth_subject, role)` - Create user with internal UUID
- `get_user_by_auth(auth_provider, auth_subject)` - Look up user by provider credentials
- `get_user(user_id)` - Get user by internal ID
- `get_tenant_by_user_id(user_id)` - Get tenant for a user

**Updated Tenant Methods:**
- `create_tenant(tenant_id, user_id, stripe_customer_id)` - Now requires `user_id` instead of `email`
- `get_tenant(tenant_id)` - Now JOINs with users table to get email
- `get_tenant_by_stripe_customer()` - Updated to JOIN with users
- `get_tenant_by_stripe_subscription()` - Updated to JOIN with users

### 3. Session Claims Contract

**Standardized Contract:**
```python
class SessionClaims(BaseModel):
    user_id: str        # Internal UUID (stable)
    tenant_id: str      # Tenant UUID
    email: str
    role: str           # 'user', 'admin'
    plan: str           # 'starter', 'pro', 'enterprise'
    status: str         # 'active', 'trial', 'past_due', etc.
```

**All auth providers must yield this contract.**

### 4. Auth Endpoints

**`POST /auth/signup`:**
- Creates or links auth provider account
- Creates internal user record with UUID
- Creates tenant linked to user
- Returns `user_id`, `tenant_id`, `session_token`
- Auth provider agnostic (works with Clerk, Supabase, etc.)

**`POST /auth/session`:**
- Validates auth provider token
- Returns standardized `SessionClaims`
- Single contract point for all providers

**`validate_clerk_token()` Function:**
- Placeholder implementation
- Shows the contract that must be implemented
- Easy to swap for `validate_supabase_token()` later

### 5. Migration Guide

Created `docs/AUTH_MIGRATION_GUIDE.md` with:
- Core principles
- Database schema details
- Implementation examples
- Step-by-step migration instructions
- Benefits explanation

---

## 🔧 What Needs to Be Completed

### 1. Implement Clerk Token Validation

**File:** `edon_gateway/main.py` → `validate_clerk_token()`

**Current:** Placeholder that returns `None`

**Needs:**
```python
def validate_clerk_token(clerk_token: str) -> Optional[SessionClaims]:
    # 1. Verify Clerk JWT signature
    import jwt
    from clerk_sdk import Clerk
    
    clerk = Clerk(api_key=CLERK_SECRET_KEY)
    decoded = clerk.verify_token(clerk_token)
    clerk_user_id = decoded["sub"]
    
    # 2. Look up user in database
    from .persistence import get_db
    db = get_db()
    user = db.get_user_by_auth("clerk", clerk_user_id)
    if not user:
        return None
    
    # 3. Get tenant for user
    tenant = db.get_tenant_by_user_id(user["id"])
    if not tenant:
        return None
    
    # 4. Return SessionClaims
    return SessionClaims(
        user_id=user["id"],
        tenant_id=tenant["id"],
        email=user["email"],
        role=user["role"],
        plan=tenant["plan"],
        status=tenant["status"]
    )
```

### 2. Update Frontend to Use `/auth/signup`

**Current:** Frontend calls `/billing/signup` (legacy)

**Needs:**
- After Clerk signup succeeds, call `/auth/signup` with:
  ```json
  {
    "auth_provider": "clerk",
    "auth_subject": "<clerk_user_id>",
    "email": "<user_email>"
  }
  ```
- Store returned `user_id` and `tenant_id` in Clerk's `publicMetadata`

### 3. Update Protected Routes to Use Session Claims

**Current:** Uses Clerk's `useAuth()` directly

**Needs:**
- Call `/auth/session` endpoint with Clerk token
- Use returned `SessionClaims` for authorization
- Check `claims.status` for subscription checks

### 4. Database Migration (If Existing Data)

**If you have existing tenants without users:**

```sql
-- Create users for existing tenants
INSERT INTO users (id, email, auth_provider, auth_subject, role, created_at, updated_at)
SELECT 
    LOWER(HEX(RANDOMBLOB(16))),  -- Generate UUID
    email,
    'legacy',
    'legacy_' || id,
    'user',
    created_at,
    updated_at
FROM tenants
WHERE user_id IS NULL;

-- Update tenants to reference users
UPDATE tenants t
SET user_id = (
    SELECT u.id 
    FROM users u 
    WHERE u.email = t.email 
    AND u.auth_provider = 'legacy'
    LIMIT 1
)
WHERE user_id IS NULL;

-- Add NOT NULL constraint after migration
-- ALTER TABLE tenants ADD COLUMN user_id_temp TEXT;
-- UPDATE tenants SET user_id_temp = user_id;
-- ALTER TABLE tenants DROP COLUMN user_id;
-- ALTER TABLE tenants ADD COLUMN user_id TEXT NOT NULL;
-- UPDATE tenants SET user_id = user_id_temp;
-- ALTER TABLE tenants DROP COLUMN user_id_temp;
```

---

## 📋 Migration Checklist

### Phase 1: Setup (Current)
- [x] Database schema updated (users table)
- [x] Database methods updated
- [x] Session claims contract defined
- [x] Auth endpoints created
- [x] Migration guide written

### Phase 2: Clerk Integration
- [ ] Install Clerk SDK: `pip install clerk-sdk-python`
- [ ] Implement `validate_clerk_token()` function
- [ ] Update frontend to call `/auth/signup` after Clerk signup
- [ ] Store `user_id`/`tenant_id` in Clerk `publicMetadata`
- [ ] Update frontend to use `/auth/session` for protected routes

### Phase 3: Testing
- [ ] Test Clerk signup flow
- [ ] Test session validation
- [ ] Test protected routes
- [ ] Test subscription checks

### Phase 4: Future Supabase Migration
- [ ] Implement `validate_supabase_token()` function
- [ ] Update `auth_provider` values in database
- [ ] Update frontend to use Supabase components
- [ ] Test migration

---

## 🎯 Benefits Achieved

### 1. Stable User IDs
- `user_id` never changes
- Foreign keys remain valid
- No data migration needed when switching providers

### 2. Easy Provider Switching
- Change token validation function
- Update `auth_provider`/`auth_subject` values
- Done!

### 3. Authorization Stays in Your Control
- Auth provider = identity only
- EDON backend = permissions + policy
- No vendor lock-in

### 4. Standardized Contract
- All providers yield `SessionClaims`
- App logic uses claims, not provider-specific data
- Consistent authorization checks

---

## 📝 Example Usage

### Current Flow (Clerk):

```python
# 1. User signs up via Clerk (frontend)
# 2. Frontend calls /auth/signup
POST /auth/signup
{
    "auth_provider": "clerk",
    "auth_subject": "user_2abc123",
    "email": "user@example.com"
}

# Returns:
{
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "tenant_id": "tenant_a1b2c3d4",
    "session_token": "edon_session_..."
}

# 3. Frontend stores user_id/tenant_id in Clerk metadata
# 4. For protected routes, validate token:
POST /auth/session
Authorization: Bearer <clerk_token>

# Returns SessionClaims
{
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "tenant_id": "tenant_a1b2c3d4",
    "email": "user@example.com",
    "role": "user",
    "plan": "starter",
    "status": "active"
}
```

### Future Flow (Supabase):

```python
# Same endpoints, different token validation:
# validate_supabase_token() instead of validate_clerk_token()
# Everything else stays the same!
```

---

*Last Updated: 2025-01-27*
