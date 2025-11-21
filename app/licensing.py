"""
EDON v2.0.0 - License Enforcement Module

Enforces evaluation license terms:
- Cloud activation + 30-day limit
- Periodic validation
- Revocation support
- Source code protection
- Clock skew handling
- Key rotation support
- Per-OEM/project tracking
- GDPR-compliant data transmission
"""

import os
import json
import time
import hashlib
import hmac
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import requests

logger = logging.getLogger(__name__)

# License configuration
LICENSE_FILE = Path.home() / ".edon" / "license.json"
ACTIVATION_SERVER = os.getenv("EDON_ACTIVATION_SERVER", "https://activation.edon.ai/v1/validate")
EVAL_PERIOD_DAYS = 30
VALIDATION_INTERVAL = 3600  # 1 hour
OFFLINE_GRACE_PERIOD = 86400  # 24 hours offline allowed
CLOCK_SKEW_TOLERANCE = 21600  # 6 hours tolerance for clock skew

# HMAC key versions (for key rotation)
# In production, these should be stored server-side only
# This is a fallback for offline/dev scenarios
LICENSE_SECRETS = {
    "v1": os.getenv("EDON_LICENSE_SECRET_V1", "eval-secret-key-v1-change-in-production"),
    "v2": os.getenv("EDON_LICENSE_SECRET_V2", "eval-secret-key-v2-change-in-production"),
}
CURRENT_KEY_VERSION = "v1"  # Current key version for new licenses
MIN_SUPPORTED_KEY_VERSION = "v1"  # Oldest key version still accepted


class LicenseError(Exception):
    """License validation error."""
    pass


class LicenseValidator:
    """Validates and enforces EDON licenses."""
    
    def __init__(self):
        self.license_data: Optional[Dict[str, Any]] = None
        self.last_validation: float = 0
        self.offline_since: Optional[float] = None
        self.server_time_offset: Optional[float] = None  # Clock skew compensation
        self._load_license()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup structured logging for license operations."""
        # License operations are logged at INFO level
        # Failures are logged at WARNING/ERROR level
        pass  # Using module-level logger
    
    def _log_activation_attempt(self, success: bool, reason: str = "", **kwargs):
        """Log activation attempt with structured data."""
        log_data = {
            "event": "license_activation",
            "success": success,
            "reason": reason,
            "timestamp": time.time(),
            **kwargs
        }
        if success:
            logger.info(f"[LICENSE] Activation successful: {log_data}")
        else:
            logger.warning(f"[LICENSE] Activation failed: {log_data}")
    
    def _log_validation(self, success: bool, reason: str = "", **kwargs):
        """Log validation attempt with structured data."""
        log_data = {
            "event": "license_validation",
            "success": success,
            "reason": reason,
            "timestamp": time.time(),
            **kwargs
        }
        if success:
            logger.debug(f"[LICENSE] Validation successful: {log_data}")
        else:
            logger.warning(f"[LICENSE] Validation failed: {log_data}")
    
    def _log_rejection(self, reason: str, **kwargs):
        """Log license rejection with structured data."""
        log_data = {
            "event": "license_rejection",
            "reason": reason,
            "timestamp": time.time(),
            **kwargs
        }
        logger.warning(f"[LICENSE] Request rejected: {log_data}")
    
    def _load_license(self):
        """Load license from file."""
        if LICENSE_FILE.exists():
            try:
                with open(LICENSE_FILE, 'r') as f:
                    self.license_data = json.load(f)
                logger.info(f"[LICENSE] Loaded license from {LICENSE_FILE}")
            except Exception as e:
                logger.warning(f"[LICENSE] Failed to load license: {e}")
                self.license_data = None
        else:
            logger.info("[LICENSE] No license file found, will attempt activation")
    
    def _save_license(self, data: Dict[str, Any]):
        """Save license to file."""
        LICENSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LICENSE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        self.license_data = data
        logger.info(f"[LICENSE] Saved license to {LICENSE_FILE}")
    
    def _get_secret_key(self, key_version: str) -> bytes:
        """Get HMAC secret key for given version."""
        if key_version not in LICENSE_SECRETS:
            raise LicenseError(f"Unsupported key version: {key_version}")
        return LICENSE_SECRETS[key_version].encode()
    
    def _compute_signature(self, data: Dict[str, Any], key_version: Optional[str] = None) -> str:
        """Compute HMAC signature for license data."""
        if key_version is None:
            key_version = data.get("key_version", CURRENT_KEY_VERSION)
        
        secret_key = self._get_secret_key(key_version)
        
        # Create canonical representation (exclude signature field)
        data_copy = {k: v for k, v in data.items() if k != "signature"}
        canonical = json.dumps(data_copy, sort_keys=True, separators=(',', ':'))
        return hmac.new(secret_key, canonical.encode(), hashlib.sha256).hexdigest()
    
    def _verify_signature(self, data: Dict[str, Any], signature: str) -> bool:
        """Verify license signature with key rotation support."""
        key_version = data.get("key_version", "v1")  # Default to v1 for backwards compatibility
        
        # Check if key version is still supported
        if key_version < MIN_SUPPORTED_KEY_VERSION:
            self._log_rejection("unsupported_key_version", key_version=key_version)
            return False
        
        expected = self._compute_signature(data, key_version)
        is_valid = hmac.compare_digest(expected, signature)
        
        if not is_valid:
            self._log_rejection("invalid_signature", key_version=key_version)
        
        return is_valid
    
    def _handle_clock_skew(self, server_time: float, local_time: float) -> float:
        """
        Handle clock skew between server and client.
        
        Returns:
            Adjusted local time (compensated for skew)
        """
        offset = server_time - local_time
        
        # If offset is within tolerance, use it
        if abs(offset) <= CLOCK_SKEW_TOLERANCE:
            if self.server_time_offset is None:
                self.server_time_offset = offset
                logger.info(f"[LICENSE] Clock skew detected: {offset:.1f}s (within tolerance)")
            else:
                # Average with previous offset (smooth changes)
                self.server_time_offset = (self.server_time_offset * 0.7) + (offset * 0.3)
            return local_time + self.server_time_offset
        
        # If offset is too large, log warning but don't adjust
        logger.warning(f"[LICENSE] Large clock skew detected: {offset:.1f}s (exceeds {CLOCK_SKEW_TOLERANCE}s tolerance)")
        return local_time
    
    def _get_adjusted_time(self) -> float:
        """Get current time adjusted for clock skew."""
        local_time = time.time()
        if self.server_time_offset is not None:
            return local_time + self.server_time_offset
        return local_time
    
    def activate(self, activation_code: Optional[str] = None, 
                 org_id: Optional[str] = None,
                 project_id: Optional[str] = None) -> bool:
        """
        Activate evaluation license.
        
        Args:
            activation_code: Optional activation code (for future use)
            org_id: Optional organization ID
            project_id: Optional project ID
        
        Returns:
            True if activation successful
        """
        try:
            # Try cloud activation
            activation_data = {
                "version": "2.0.0",
                "type": "evaluation",
                "timestamp": time.time(),
                "hostname": os.getenv("HOSTNAME", "unknown"),
                "activation_code": activation_code,
                "org_id": org_id,
                "project_id": project_id
            }
            
            # GDPR: Only send minimal anonymized data
            # Do NOT send: raw sensor data, personal info, user IDs
            gdpr_safe_data = {
                "version": activation_data["version"],
                "type": activation_data["type"],
                "timestamp": activation_data["timestamp"],
                "hostname_hash": hashlib.sha256(activation_data["hostname"].encode()).hexdigest()[:16],  # Anonymized
                "activation_code": activation_code,
                "org_id": org_id,  # Organization ID (not personal)
                "project_id": project_id  # Project ID (not personal)
            }
            
            try:
                response = requests.post(
                    f"{ACTIVATION_SERVER}/activate",
                    json=gdpr_safe_data,
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("ok"):
                        # Get server time for clock skew handling
                        server_time = result.get("server_time", time.time())
                        self._handle_clock_skew(server_time, time.time())
                        
                        # Create license
                        license_data = {
                            "type": "evaluation",
                            "version": "2.0.0",
                            "key_version": CURRENT_KEY_VERSION,
                            "activated_at": time.time(),
                            "expires_at": time.time() + (EVAL_PERIOD_DAYS * 86400),
                            "activation_id": result.get("activation_id", "local"),
                            "org_id": org_id or result.get("org_id"),
                            "project_id": project_id or result.get("project_id"),
                            "plan": result.get("plan", "evaluation"),
                            "hostname": activation_data["hostname"],
                            "revoked": False
                        }
                        license_data["signature"] = self._compute_signature(license_data)
                        self._save_license(license_data)
                        self._log_activation_attempt(True, activation_id=license_data["activation_id"])
                        logger.info("[LICENSE] Cloud activation successful")
                        return True
            except requests.exceptions.RequestException as e:
                logger.warning(f"[LICENSE] Cloud activation failed: {e}, using offline activation")
            
            # Fallback: Offline activation (for development/testing)
            # In production, this would be disabled
            if os.getenv("EDON_ALLOW_OFFLINE_ACTIVATION", "false").lower() == "true":
                logger.warning("[LICENSE] Using offline activation (development mode)")
                license_data = {
                    "type": "evaluation",
                    "version": "2.0.0",
                    "key_version": CURRENT_KEY_VERSION,
                    "activated_at": time.time(),
                    "expires_at": time.time() + (EVAL_PERIOD_DAYS * 86400),
                    "activation_id": "offline-dev",
                    "org_id": org_id or "dev-org",
                    "project_id": project_id or "dev-project",
                    "plan": "evaluation",
                    "hostname": activation_data["hostname"],
                    "revoked": False,
                    "offline": True
                }
                license_data["signature"] = self._compute_signature(license_data)
                self._save_license(license_data)
                self._log_activation_attempt(True, activation_id="offline-dev", offline=True)
                return True
            
            self._log_activation_attempt(False, reason="cloud_unavailable_and_offline_disabled")
            raise LicenseError("Activation failed: cloud server unavailable and offline activation disabled")
            
        except Exception as e:
            logger.error(f"[LICENSE] Activation error: {e}")
            self._log_activation_attempt(False, reason=str(e))
            raise LicenseError(f"Activation failed: {e}")
    
    def validate(self, force_online: bool = False) -> bool:
        """
        Validate current license.
        
        Args:
            force_online: Force online validation (skip cache)
        
        Returns:
            True if license is valid
        """
        # Check if validation needed
        now = self._get_adjusted_time()
        if not force_online and (now - self.last_validation) < VALIDATION_INTERVAL:
            return True  # Use cached validation
        
        # Load license if not loaded
        if self.license_data is None:
            self._load_license()
        
        # No license = need activation
        if self.license_data is None:
            logger.warning("[LICENSE] No license found, attempting activation...")
            self._log_validation(False, reason="no_license")
            try:
                return self.activate()
            except Exception as e:
                logger.error(f"[LICENSE] Auto-activation failed: {e}")
                self._log_validation(False, reason=f"auto_activation_failed: {e}")
                raise LicenseError("License not activated. Please activate your evaluation license.")
        
        # Check signature
        signature = self.license_data.pop("signature", None)
        if signature is None or not self._verify_signature(self.license_data, signature):
            self._log_rejection("invalid_signature", 
                              activation_id=self.license_data.get("activation_id"))
            raise LicenseError("License validation failed: invalid signature")
        self.license_data["signature"] = signature  # Restore
        
        # Check if revoked
        if self.license_data.get("revoked", False):
            self._log_rejection("revoked", 
                              activation_id=self.license_data.get("activation_id"))
            raise LicenseError("License has been revoked. Please contact support@edon.ai")
        
        # Check expiration (with clock skew tolerance)
        expires_at = self.license_data.get("expires_at", 0)
        if now > expires_at:
            # Check if expiration is within clock skew tolerance
            if (now - expires_at) > CLOCK_SKEW_TOLERANCE:
                self._log_rejection("expired", 
                                  activation_id=self.license_data.get("activation_id"),
                                  expires_at=expires_at,
                                  current_time=now)
                raise LicenseError(
                    f"License expired. Evaluation period ended. "
                    f"Contact licensing@edon.ai for production licensing."
                )
            else:
                # Within clock skew tolerance - log warning but allow
                logger.warning(f"[LICENSE] License appears expired but within clock skew tolerance")
        
        # Online validation (if not offline mode)
        if not self.license_data.get("offline", False):
            try:
                # GDPR: Only send minimal anonymized data
                validation_data = {
                    "activation_id": self.license_data.get("activation_id"),
                    "hostname_hash": hashlib.sha256(
                        self.license_data.get("hostname", "unknown").encode()
                    ).hexdigest()[:16],  # Anonymized
                    "org_id": self.license_data.get("org_id"),
                    "project_id": self.license_data.get("project_id"),
                    "timestamp": now
                }
                
                response = requests.post(
                    f"{ACTIVATION_SERVER}/validate",
                    json=validation_data,
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if not result.get("ok"):
                        # License revoked or invalid
                        self.license_data["revoked"] = True
                        self._save_license(self.license_data)
                        self._log_rejection("revoked_by_server", 
                                          activation_id=self.license_data.get("activation_id"))
                        raise LicenseError("License has been revoked by server")
                    
                    # Handle clock skew
                    server_time = result.get("server_time")
                    if server_time:
                        self._handle_clock_skew(server_time, time.time())
                    
                    # Update last validation
                    self.last_validation = now
                    self.offline_since = None
                    self._log_validation(True, activation_id=self.license_data.get("activation_id"))
                    return True
                else:
                    # Server error - allow offline grace period
                    if self.offline_since is None:
                        self.offline_since = now
                    elif (now - self.offline_since) > OFFLINE_GRACE_PERIOD:
                        self._log_rejection("server_unreachable_extended", 
                                          activation_id=self.license_data.get("activation_id"))
                        raise LicenseError(
                            "License validation failed: server unreachable for extended period. "
                            "Please check your internet connection."
                        )
                    logger.warning("[LICENSE] Server unreachable, using offline grace period")
                    self._log_validation(True, reason="offline_grace_period", 
                                       activation_id=self.license_data.get("activation_id"))
                    return True
                    
            except requests.exceptions.RequestException as e:
                # Network error - allow offline grace period
                if self.offline_since is None:
                    self.offline_since = now
                elif (now - self.offline_since) > OFFLINE_GRACE_PERIOD:
                    self._log_rejection("network_error_extended", 
                                      activation_id=self.license_data.get("activation_id"),
                                      error=str(e))
                    raise LicenseError(
                        "License validation failed: cannot reach activation server. "
                        "Please check your internet connection."
                    )
                logger.warning(f"[LICENSE] Network error: {e}, using offline grace period")
                self._log_validation(True, reason="offline_grace_period", 
                                   activation_id=self.license_data.get("activation_id"))
                return True
        else:
            # Offline mode - just check expiration
            self.last_validation = now
            self._log_validation(True, reason="offline_mode", 
                               activation_id=self.license_data.get("activation_id"))
            return True
    
    def get_license_info(self) -> Dict[str, Any]:
        """Get license information."""
        if self.license_data is None:
            return {"status": "not_activated"}
        
        expires_at = self.license_data.get("expires_at", 0)
        now = self._get_adjusted_time()
        days_remaining = max(0, int((expires_at - now) / 86400))
        
        return {
            "status": "active" if not self.license_data.get("revoked") else "revoked",
            "type": self.license_data.get("type", "unknown"),
            "version": self.license_data.get("version", "unknown"),
            "key_version": self.license_data.get("key_version", "v1"),
            "activated_at": datetime.fromtimestamp(self.license_data.get("activated_at", 0)).isoformat(),
            "expires_at": datetime.fromtimestamp(expires_at).isoformat(),
            "days_remaining": days_remaining,
            "offline": self.license_data.get("offline", False),
            "revoked": self.license_data.get("revoked", False),
            "org_id": self.license_data.get("org_id"),
            "project_id": self.license_data.get("project_id"),
            "plan": self.license_data.get("plan", "evaluation"),
            "activation_id": self.license_data.get("activation_id")
        }


# Global validator instance
_validator: Optional[LicenseValidator] = None


def get_validator() -> LicenseValidator:
    """Get global license validator instance."""
    global _validator
    if _validator is None:
        _validator = LicenseValidator()
    return _validator


def validate_license(force_online: bool = False) -> bool:
    """
    Validate license (convenience function).
    
    Raises:
        LicenseError: If license is invalid
    """
    return get_validator().validate(force_online=force_online)


def get_license_info() -> Dict[str, Any]:
    """Get license information."""
    return get_validator().get_license_info()
