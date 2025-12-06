"""
Robot Stability Adaptive Memory Engine

Learns patterns from robot stability interventions to adaptively adjust
strategy selection and modulations for better performance.

Algorithm:
1. Record each intervention event (strategy, modulations, outcome)
2. Track success rates per strategy/modulation combination
3. Compute adaptive adjustments to modulations based on learned patterns
4. Personalize to specific environment (e.g., MuJoCo) over time
"""

import sqlite3
import math
import statistics
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np


class RobotStabilityMemory:
    """
    Adaptive Memory Engine for Robot Stability.
    
    Maintains rolling history of interventions and learns which strategies
    and modulations work best in the current environment.
    """
    
    def __init__(self, db_path: str = "data/robot_stability_memory.db", window_size: int = 1000):
        """
        Initialize the Robot Stability Memory Engine.
        
        Args:
            db_path: Path to SQLite database for persistence
            window_size: Rolling window size (number of recent interventions to track)
        """
        self.db_path = Path(db_path)
        self.window_size = window_size
        
        # In-memory rolling buffer (stores last N intervention events)
        # Each record: (timestamp, strategy_id, modulations, intervention_occurred, fail_risk)
        self.buffer: deque = deque(maxlen=window_size)
        
        # Strategy success tracking
        # strategy_id -> {success_count, total_count, success_rate}
        self.strategy_stats: Dict[int, Dict] = defaultdict(lambda: {
            'success_count': 0,
            'total_count': 0,
            'success_rate': 0.5  # Default 50% (unknown)
        })
        
        # Adaptive baselines (EWMA - continuously adapting)
        # Strategy success rate baselines (rolling EWMA)
        self.strategy_baselines: Dict[int, Dict[str, float]] = defaultdict(lambda: {
            'success_rate_mu': 0.5,  # EWMA mean success rate
            'success_rate_var': 0.25,  # EWMA variance
            'count': 0
        })
        
        # Modulation pattern tracking
        # Track which modulation ranges work best
        self.modulation_patterns: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
        # gain_scale, lateral_compliance, step_height_bias -> success/failure
        
        # Intervention risk baseline (EWMA - continuously adapting)
        self.risk_baseline: Dict[str, float] = {
            'mu': 0.5,  # EWMA mean intervention risk
            'std': 0.2,  # EWMA standard deviation
            'count': 0
        }
        
        # Overall success rate baseline (EWMA)
        self.success_baseline: Dict[str, float] = {
            'mu': 0.5,  # EWMA mean success rate
            'std': 0.2,  # EWMA standard deviation
            'count': 0
        }
        
        # Initialize database
        self._init_database()
        
        # Load recent history from database
        self._load_recent_history()
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS robot_stability_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                strategy_id INTEGER NOT NULL,
                gain_scale REAL NOT NULL,
                lateral_compliance REAL NOT NULL,
                step_height_bias REAL NOT NULL,
                intervention_occurred INTEGER NOT NULL,
                fail_risk REAL NOT NULL,
                roll REAL,
                pitch REAL,
                roll_velocity REAL,
                pitch_velocity REAL
            )
        """)
        
        # Index for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON robot_stability_memory(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_strategy ON robot_stability_memory(strategy_id)
        """)
        
        conn.commit()
        conn.close()
    
    def _load_recent_history(self):
        """Load recent history from database into memory buffer."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Load last N records
        cursor.execute("""
            SELECT timestamp, strategy_id, gain_scale, lateral_compliance, 
                   step_height_bias, intervention_occurred, fail_risk
            FROM robot_stability_memory
            ORDER BY timestamp DESC
            LIMIT ?
        """, (self.window_size,))
        
        records = []
        for row in cursor.fetchall():
            record = {
                'timestamp': row[0],
                'strategy_id': row[1],
                'modulations': {
                    'gain_scale': row[2],
                    'lateral_compliance': row[3],
                    'step_height_bias': row[4]
                },
                'intervention_occurred': bool(row[5]),
                'fail_risk': row[6]
            }
            records.append(record)
        
        conn.close()
        
        # Add to buffer (oldest first)
        for record in reversed(records):
            self.buffer.append(record)
        
        # Recompute statistics
        self._update_statistics()
    
    def record_intervention(
        self,
        strategy_id: int,
        modulations: Dict[str, float],
        intervention_occurred: bool,
        fail_risk: float,
        robot_state: Optional[Dict[str, float]] = None
    ):
        """
        Record an intervention event (or successful step).
        
        Args:
            strategy_id: Strategy ID (0-3)
            modulations: Dictionary with gain_scale, lateral_compliance, step_height_bias
            intervention_occurred: True if intervention happened, False if stable
            fail_risk: Intervention risk from fail-risk model
            robot_state: Optional robot state (roll, pitch, velocities) for context
        """
        timestamp = datetime.now().timestamp()
        
        # Create record
        record = {
            'timestamp': timestamp,
            'strategy_id': strategy_id,
            'modulations': modulations.copy(),
            'intervention_occurred': intervention_occurred,
            'fail_risk': fail_risk
        }
        
        # Add to in-memory buffer
        self.buffer.append(record)
        
        # Persist to database (non-blocking - batch writes)
        # Only write to DB every N records to avoid blocking
        # This prevents database writes from slowing down the API
        if len(self.buffer) % 10 == 0:  # Write every 10 records
            try:
                conn = sqlite3.connect(str(self.db_path), timeout=0.1)  # Short timeout
                cursor = conn.cursor()
                
                robot_state = robot_state or {}
                cursor.execute("""
                    INSERT INTO robot_stability_memory 
                    (timestamp, strategy_id, gain_scale, lateral_compliance, step_height_bias,
                     intervention_occurred, fail_risk, roll, pitch, roll_velocity, pitch_velocity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp, strategy_id,
                    modulations.get('gain_scale', 1.0),
                    modulations.get('lateral_compliance', 1.0),
                    modulations.get('step_height_bias', 0.0),
                    1 if intervention_occurred else 0,
                    fail_risk,
                    robot_state.get('roll', 0.0),
                    robot_state.get('pitch', 0.0),
                    robot_state.get('roll_velocity', 0.0),
                    robot_state.get('pitch_velocity', 0.0)
                ))
                
                conn.commit()
                conn.close()
            except Exception:
                # Silently fail - DB writes are best-effort, don't block API
                pass
        
        # Update statistics more frequently for faster learning
        # But not every step (too expensive)
        if len(self.buffer) % 20 == 0:
            self._update_statistics()
    
    def _update_statistics(self):
        """Update strategy success rates and modulation patterns."""
        # Reset strategy stats
        for strategy_id in range(4):  # 4 strategies
            self.strategy_stats[strategy_id] = {
                'success_count': 0,
                'total_count': 0,
                'success_rate': 0.5
            }
        
        # Count successes per strategy
        for record in self.buffer:
            strategy_id = record['strategy_id']
            success = not record['intervention_occurred']  # Success = no intervention
            
            if strategy_id not in self.strategy_stats:
                self.strategy_stats[strategy_id] = {
                    'success_count': 0,
                    'total_count': 0,
                    'success_rate': 0.5
                }
            
            self.strategy_stats[strategy_id]['total_count'] += 1
            if success:
                self.strategy_stats[strategy_id]['success_count'] += 1
        
        # Compute success rates
        for strategy_id, stats in self.strategy_stats.items():
            if stats['total_count'] > 0:
                stats['success_rate'] = stats['success_count'] / stats['total_count']
        
        # Update adaptive baselines using EWMA (continuously adapting)
        alpha = 0.3  # EWMA smoothing factor (same as CAV adaptive memory)
        
        # 1. Update strategy success rate baselines (EWMA per strategy)
        for strategy_id in range(4):  # 4 strategies
            strategy_records = [r for r in self.buffer if r['strategy_id'] == strategy_id]
            if len(strategy_records) >= 10:  # Need some data
                recent_successes = [
                    1.0 if not r['intervention_occurred'] else 0.0
                    for r in strategy_records[-50:]  # Last 50 for this strategy
                ]
                current_mean = statistics.mean(recent_successes)
                current_var = statistics.variance(recent_successes) if len(recent_successes) > 1 else 0.25
                
                baseline = self.strategy_baselines[strategy_id]
                baseline['success_rate_mu'] = (
                    alpha * current_mean + (1 - alpha) * baseline['success_rate_mu']
                )
                baseline['success_rate_var'] = (
                    alpha * current_var + (1 - alpha) * baseline['success_rate_var']
                )
                baseline['count'] = len(strategy_records)
        
        # 2. Update overall risk baseline (EWMA)
        if self.buffer:
            recent_risks = [r['fail_risk'] for r in list(self.buffer)[-100:]]  # Last 100
            current_mean = statistics.mean(recent_risks)
            current_std = statistics.stdev(recent_risks) if len(recent_risks) > 1 else 0.2
            
            self.risk_baseline['mu'] = (
                alpha * current_mean + (1 - alpha) * self.risk_baseline['mu']
            )
            self.risk_baseline['std'] = (
                alpha * current_std + (1 - alpha) * self.risk_baseline['std']
            )
            self.risk_baseline['count'] = len(self.buffer)
        
        # 3. Update overall success rate baseline (EWMA)
        if self.buffer:
            recent_successes = [
                1.0 if not r['intervention_occurred'] else 0.0
                for r in list(self.buffer)[-100:]  # Last 100
            ]
            current_mean = statistics.mean(recent_successes)
            current_std = statistics.stdev(recent_successes) if len(recent_successes) > 1 else 0.2
            
            self.success_baseline['mu'] = (
                alpha * current_mean + (1 - alpha) * self.success_baseline['mu']
            )
            self.success_baseline['std'] = (
                alpha * current_std + (1 - alpha) * self.success_baseline['std']
            )
            self.success_baseline['count'] = len(self.buffer)
    
    def compute_adaptive_modulations(
        self,
        strategy_id: int,
        base_modulations: Dict[str, float],
        fail_risk: float,
        robot_state: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Compute adaptive adjustments to modulations based on learned patterns.
        
        Args:
            strategy_id: Strategy ID from v8 policy
            base_modulations: Base modulations from v8 policy
            fail_risk: Current intervention risk
            robot_state: Optional robot state for context
        
        Returns:
            Adjusted modulations dictionary
        """
        # Start with base modulations
        adjusted = base_modulations.copy()
        
        # CRITICAL: Need significant history before making adjustments
        # Early adjustments based on noisy data can make things worse
        # Increased to 500 records to be VERY conservative for consistent demo performance
        # This ensures adaptive memory only activates after substantial learning
        if len(self.buffer) < 500:
            return adjusted  # Not enough data - use base modulations (consistent performance)
        
        # Update statistics if needed (ensure we have fresh data)
        # Reduced frequency to avoid blocking - update every 50 records instead of 20
        if len(self.buffer) % 50 == 0:
            self._update_statistics()
        
        # 1. Strategy-based adjustment using adaptive baseline (EWMA)
        # Use adaptive baseline (continuously moving) instead of raw success rate
        strategy_baseline = self.strategy_baselines.get(strategy_id, {
            'success_rate_mu': 0.5,
            'success_rate_var': 0.25,
            'count': 0
        })
        
        # Need at least 100 samples per strategy for reliable baseline (very conservative)
        # Increased for more consistent performance
        if strategy_baseline['count'] >= 100:
            # Use adaptive baseline (EWMA mean) - this is continuously adapting
            baseline_success_rate = strategy_baseline['success_rate_mu']
            baseline_std = math.sqrt(strategy_baseline['success_rate_var'])
            
            # Compute z-score: how far is current strategy from its adaptive baseline?
            # Get current success rate
            strategy_stats = self.strategy_stats.get(strategy_id, {
                'success_rate': 0.5,
                'total_count': 0
            })
            current_success_rate = strategy_stats.get('success_rate', 0.5)
            
            # Compute z-score relative to adaptive baseline
            if baseline_std > 0.1:  # Need meaningful std
                z_score = (current_success_rate - baseline_success_rate) / baseline_std
            else:
                z_score = 0.0
            
            # Adjust based on deviation from adaptive baseline
            # Made MUCH more conservative: require z < -2.0 (was -1.0) for worse performance
            # And z > 2.0 (was 1.0) for better performance
            # Adjustments are also smaller
            if z_score < -2.0:
                # Very small reduction (0.98x) - was 0.95x
                adjusted['gain_scale'] = base_modulations['gain_scale'] * 0.98
                adjusted['lateral_compliance'] = min(
                    base_modulations['lateral_compliance'] * 1.02, 1.0  # Was 1.05
                )
            # If significantly above baseline (z > 2.0), can be slightly more aggressive
            elif z_score > 2.0:
                # Very small increase (1.01x) - was 1.02x
                adjusted['gain_scale'] = min(
                    base_modulations['gain_scale'] * 1.01, 1.5
                )
            # If near baseline, use absolute thresholds as fallback (more conservative)
            elif baseline_success_rate < 0.30:  # Was 0.35
                # Baseline itself is very low - make very small conservative adjustment
                adjusted['gain_scale'] = base_modulations['gain_scale'] * 0.99  # Was 0.97
            elif baseline_success_rate > 0.80:  # Was 0.75
                # Baseline itself is very high - can be slightly aggressive
                adjusted['gain_scale'] = min(
                    base_modulations['gain_scale'] * 1.005, 1.5  # Was 1.01
                )
        # If we don't have enough data for this strategy, don't adjust
        
        # 2. Risk-based adjustment using adaptive baseline (EWMA - continuously adapting)
        # Need at least 200 records for reliable risk baseline (very conservative)
        if self.risk_baseline['count'] >= 200:
            # Use adaptive baseline (EWMA mean) - this is continuously adapting
            risk_z_score = 0.0
            if self.risk_baseline['std'] > 0.1:  # Need meaningful std
                # Compute z-score: how far is current risk from adaptive baseline?
                risk_z_score = (fail_risk - self.risk_baseline['mu']) / self.risk_baseline['std']
                
                # More conservative thresholds (z > 2.0 instead of 1.5)
                # High risk (z > 2.0): Very small conservative adjustment
                if risk_z_score > 2.0:
                    adjusted['gain_scale'] = adjusted['gain_scale'] * 0.99  # Was 0.97
                    adjusted['lateral_compliance'] = min(adjusted['lateral_compliance'] * 1.01, 1.0)  # Was 1.03
                # Low risk (z < -2.0): Very small aggressive adjustment
                elif risk_z_score < -2.0:
                    adjusted['gain_scale'] = min(adjusted['gain_scale'] * 1.005, 1.5)  # Was 1.01
        
        # 3. Robot state-based adjustment (always safe - based on current state, not history)
        if robot_state:
            roll = abs(robot_state.get('roll', 0.0))
            pitch = abs(robot_state.get('pitch', 0.0))
            
            # If robot is very tilted (>0.3 rad), be more aggressive
            if max(roll, pitch) > 0.3:
                adjusted['gain_scale'] = min(adjusted['gain_scale'] * 1.05, 1.5)  # Reduced from 1.1
            # If robot is stable (<0.1 rad), be slightly more conservative
            elif max(roll, pitch) < 0.1:
                adjusted['gain_scale'] = adjusted['gain_scale'] * 0.99  # Reduced from 0.98
        
        # Clamp to valid ranges
        adjusted['gain_scale'] = np.clip(adjusted['gain_scale'], 0.5, 1.5)
        adjusted['lateral_compliance'] = np.clip(adjusted['lateral_compliance'], 0.0, 1.0)
        # step_height_bias can stay as is (already clamped by v8 policy)
        
        return adjusted
    
    def get_summary(self) -> Dict:
        """
        Get summary of learned patterns.
        
        Returns:
            Dictionary with strategy stats, risk baseline, and recent patterns
        """
        return {
            'total_records': len(self.buffer),
            'strategy_stats': dict(self.strategy_stats),
            'strategy_baselines': {
                k: v.copy() for k, v in self.strategy_baselines.items()
            },
            'risk_baseline': self.risk_baseline.copy(),
            'success_baseline': self.success_baseline.copy(),
            'recent_success_rate': (
                sum(1 for r in list(self.buffer)[-100:] if not r['intervention_occurred']) / 
                min(100, len(self.buffer))
                if self.buffer else 0.5
            )
        }
    
    def clear(self):
        """Clear all memory (for testing)."""
        self.buffer.clear()
        self.strategy_stats.clear()
        self.strategy_baselines.clear()
        self.modulation_patterns.clear()
        self.risk_baseline = {
            'mu': 0.5,
            'std': 0.2,
            'count': 0
        }
        self.success_baseline = {
            'mu': 0.5,
            'std': 0.2,
            'count': 0
        }
        
        # Clear database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM robot_stability_memory")
        conn.commit()
        conn.close()


# Global instance (initialized on startup)
_robot_stability_memory: Optional[RobotStabilityMemory] = None


def get_robot_stability_memory() -> RobotStabilityMemory:
    """Get global robot stability memory instance."""
    global _robot_stability_memory
    if _robot_stability_memory is None:
        _robot_stability_memory = RobotStabilityMemory()
    return _robot_stability_memory


def set_robot_stability_memory(memory: RobotStabilityMemory):
    """Set global robot stability memory instance (for testing)."""
    global _robot_stability_memory
    _robot_stability_memory = memory

