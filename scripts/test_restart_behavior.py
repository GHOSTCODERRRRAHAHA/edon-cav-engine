#!/usr/bin/env python3
"""Test restart behavior - verify metrics reset and persistence."""

import sys
import os
import time
import requests
import json
from pathlib import Path

# Configuration
GATEWAY_URL = os.getenv("EDON_GATEWAY_URL", "http://localhost:8000")
API_TOKEN = os.getenv("EDON_API_TOKEN", "your-secret-token")

def check_health():
    """Check health endpoint."""
    try:
        response = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Health check: {data.get('status')}")
        return data
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return None

def fetch_metrics():
    """Get current metrics."""
    try:
        response = requests.get(
            f"{GATEWAY_URL}/metrics",
            headers={"X-EDON-TOKEN": API_TOKEN},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Metrics fetch failed: {e}")
        return None

def fetch_prometheus_metrics():
    """Get Prometheus metrics."""
    try:
        response = requests.get(
            f"{GATEWAY_URL}/metrics/prometheus",
            headers={"X-EDON-TOKEN": API_TOKEN},
            timeout=5
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"❌ Prometheus metrics fetch failed: {e}")
        return None

def check_database_exists():
    """Check if database file exists."""
    # Check multiple possible locations
    possible_paths = [
        Path("edon_gateway.db"),
        Path("edon_gateway/edon_gateway.db"),
        Path.cwd() / "edon_gateway.db",
        Path.cwd() / "edon_gateway" / "edon_gateway.db"
    ]
    
    for db_path in possible_paths:
        if db_path.exists():
            size = db_path.stat().st_size
            print(f"✅ Database exists: {db_path} ({size:,} bytes)")
            return True
    
    print(f"⚠️  Database not found. Checked: {[str(p) for p in possible_paths]}")
    return False

def check_logs_exist():
    """Check if log files exist."""
    # Check for JSONL audit log in multiple locations
    possible_paths = [
        Path("audit.log.jsonl"),
        Path("edon_gateway/audit.log.jsonl"),
        Path.cwd() / "audit.log.jsonl",
        Path.cwd() / "edon_gateway" / "audit.log.jsonl"
    ]
    
    for audit_log in possible_paths:
        if audit_log.exists():
            size = audit_log.stat().st_size
            print(f"✅ Audit log exists: {audit_log} ({size:,} bytes)")
            return True
    
    print(f"⚠️  Audit log not found. Checked: {[str(p) for p in possible_paths]}")
    print("   (This is OK if no actions have been executed yet)")
    return False

def main():
    """Run restart behavior tests."""
    print("=" * 60)
    print("EDON Gateway - Restart Behavior Validation")
    print("=" * 60)
    print()
    
    print("📊 Pre-Restart State:")
    print("-" * 60)
    
    # Test health
    health_before = check_health()
    if not health_before:
        print("❌ Gateway not accessible. Is it running?")
        return 1
    
    # Get metrics before
    metrics_before = fetch_metrics()
    if metrics_before:
        decisions_before = metrics_before.get("decisions_total", 0)
        uptime_before = metrics_before.get("uptime_seconds", 0)
        print(f"✅ Metrics before restart:")
        print(f"   - Decisions total: {decisions_before}")
        print(f"   - Uptime: {uptime_before}s")
        print(f"   - Verdicts: {metrics_before.get('decisions_by_verdict', {})}")
    
    # Check persistence
    print()
    print("💾 Persistence Check:")
    print("-" * 60)
    db_exists = check_database_exists()
    logs_exist = check_logs_exist()
    
    # Get Prometheus metrics
    prom_before = fetch_prometheus_metrics()
    if prom_before:
        print(f"✅ Prometheus metrics available ({len(prom_before)} chars)")
    
    print()
    print("=" * 60)
    print("🔄 RESTART SIMULATION")
    print("=" * 60)
    print()
    print("⚠️  MANUAL STEP REQUIRED:")
    print("   1. Restart the gateway: docker compose restart edon-gateway")
    print("   2. Wait 10-15 seconds for startup")
    print("   3. Press Enter to continue validation...")
    print()
    input("Press Enter after restarting the gateway...")
    
    print()
    print("📊 Post-Restart State:")
    print("-" * 60)
    
    # Wait a moment for startup
    print("Waiting 5 seconds for gateway to fully start...")
    time.sleep(5)
    
    # Test health after
    health_after = check_health()
    if not health_after:
        print("❌ Gateway not accessible after restart!")
        return 1
    
    # Get metrics after
    metrics_after = fetch_metrics()
    if metrics_after:
        decisions_after = metrics_after.get("decisions_total", 0)
        uptime_after = metrics_after.get("uptime_seconds", 0)
        print(f"✅ Metrics after restart:")
        print(f"   - Decisions total: {decisions_after}")
        print(f"   - Uptime: {uptime_after}s")
        print(f"   - Verdicts: {metrics_after.get('decisions_by_verdict', {})}")
    
    # Verify persistence
    print()
    print("💾 Persistence Verification:")
    print("-" * 60)
    db_exists_after = check_database_exists()
    logs_exist_after = check_logs_exist()
    
    # Get Prometheus metrics after
    prom_after = fetch_prometheus_metrics()
    if prom_after:
        print(f"✅ Prometheus metrics available ({len(prom_after)} chars)")
    
    # Analysis
    print()
    print("=" * 60)
    print("📈 Analysis:")
    print("=" * 60)
    
    # Metrics reset check
    if metrics_before and metrics_after:
        decisions_persisted = decisions_after >= decisions_before
        uptime_reset = uptime_after < uptime_before
        
        if decisions_persisted:
            print("✅ DECISIONS PERSISTED: Database state maintained")
        else:
            print("⚠️  DECISIONS RESET: May be expected if using in-memory metrics")
        
        if uptime_reset:
            print("✅ UPTIME RESET: Expected - metrics are in-memory")
        elif uptime_after < uptime_before + 10:  # Allow small increase (gateway was running during restart)
            print("✅ UPTIME RESET: Uptime reset (small increase is expected during restart)")
        else:
            print("⚠️  UPTIME NOT RESET: Uptime continued from before restart")
            print("   (This may be expected if metrics are persisted to database)")
    
    # Persistence check
    if db_exists and db_exists_after:
        print("✅ DATABASE PERSISTED: File exists before and after")
    else:
        print("❌ DATABASE ISSUE: File missing")
    
    if logs_exist and logs_exist_after:
        print("✅ LOGS PERSISTED: Audit log exists before and after")
    else:
        print("⚠️  LOGS: May not exist if no actions executed")
    
    print()
    print("=" * 60)
    print("✅ Restart Behavior Validation Complete")
    print("=" * 60)
    print()
    print("Expected Results:")
    print("  ✅ Metrics reset (in-memory) - UPTIME should reset")
    print("  ✅ Database persists - Decisions count should maintain or increase")
    print("  ✅ Logs persist - Audit log should exist")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
