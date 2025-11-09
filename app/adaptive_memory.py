"""
EDON Adaptive Memory Engine (Soul Layer v1)

Provides learning capability to the CAV API by maintaining rolling 24-hour
context and computing adaptive adjustments based on historical patterns.

Algorithm:
1. Store each CAV response in rolling buffer (24h) + SQLite DB
2. Compute hourly EWMA statistics (mean, variance, state distributions)
3. Calculate contextual z-scores and adaptive adjustments
4. Adjust sensitivity and environment weighting based on patterns
"""

import sqlite3
import math
import statistics
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class AdaptiveMemoryEngine:
    """
    Adaptive Memory Engine for EDON CAV API.
    
    Maintains rolling 24-hour context and computes adaptive adjustments
    based on historical CAV patterns and environmental conditions.
    """
    
    def __init__(self, db_path: str = "data/memory.db", window_hours: int = 24):
        """
        Initialize the Adaptive Memory Engine.
        
        Args:
            db_path: Path to SQLite database for persistence
            window_hours: Rolling window size in hours (default: 24)
        """
        self.db_path = Path(db_path)
        self.window_hours = window_hours
        self.window_seconds = window_hours * 3600
        
        # In-memory rolling buffer (stores last 24h of records)
        # Each record: (timestamp, cav, state, parts, temp_c, humidity, aqi, local_hour)
        self.buffer: deque = deque(maxlen=10000)  # Reasonable upper bound
        
        # Hourly statistics cache (updated periodically)
        self.hourly_stats: Dict[int, Dict] = {}  # hour -> {mu, var, state_probs}
        
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
            CREATE TABLE IF NOT EXISTS cav_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                cav_raw INTEGER NOT NULL,
                cav_smooth INTEGER NOT NULL,
                state TEXT NOT NULL,
                parts_bio REAL NOT NULL,
                parts_env REAL NOT NULL,
                parts_circadian REAL NOT NULL,
                parts_p_stress REAL NOT NULL,
                temp_c REAL NOT NULL,
                humidity REAL NOT NULL,
                aqi INTEGER NOT NULL,
                local_hour INTEGER NOT NULL
            )
        """)
        
        # Index for efficient time-based queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON cav_memory(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def _load_recent_history(self):
        """Load recent history from database into memory buffer."""
        cutoff_time = datetime.now().timestamp() - self.window_seconds
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, cav_smooth, state, parts_bio, parts_env, 
                   parts_circadian, parts_p_stress, temp_c, humidity, aqi, local_hour
            FROM cav_memory
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (cutoff_time,))
        
        for row in cursor.fetchall():
            record = {
                'timestamp': row[0],
                'cav': row[1],
                'state': row[2],
                'parts': {
                    'bio': row[3],
                    'env': row[4],
                    'circadian': row[5],
                    'p_stress': row[6]
                },
                'temp_c': row[7],
                'humidity': row[8],
                'aqi': row[9],
                'local_hour': row[10]
            }
            self.buffer.append(record)
        
        conn.close()
        
        # Recompute hourly statistics
        self._update_hourly_stats()
    
    def record(
        self,
        cav_raw: int,
        cav_smooth: int,
        state: str,
        parts: Dict[str, float],
        temp_c: float,
        humidity: float,
        aqi: int,
        local_hour: int
    ):
        """
        Record a new CAV response in memory.
        
        Args:
            cav_raw: Raw CAV score
            cav_smooth: Smoothed CAV score
            state: State string (overload, balanced, focus, restorative)
            parts: Component parts dictionary
            temp_c: Environmental temperature
            humidity: Humidity percentage
            aqi: Air Quality Index
            local_hour: Local hour [0-23]
        """
        timestamp = datetime.now().timestamp()
        
        # Create record
        record = {
            'timestamp': timestamp,
            'cav': cav_smooth,  # Use smoothed CAV for statistics
            'state': state,
            'parts': parts,
            'temp_c': temp_c,
            'humidity': humidity,
            'aqi': aqi,
            'local_hour': local_hour
        }
        
        # Add to in-memory buffer
        self.buffer.append(record)
        
        # Persist to database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO cav_memory 
            (timestamp, cav_raw, cav_smooth, state, parts_bio, parts_env, 
             parts_circadian, parts_p_stress, temp_c, humidity, aqi, local_hour)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, cav_raw, cav_smooth, state,
            parts.get('bio', 0.0), parts.get('env', 0.0),
            parts.get('circadian', 0.0), parts.get('p_stress', 0.0),
            temp_c, humidity, aqi, local_hour
        ))
        
        conn.commit()
        conn.close()
        
        # Clean old records (keep only last 7 days)
        self._cleanup_old_records()
        
        # Update hourly statistics periodically (every 10 records or hourly)
        if len(self.buffer) % 10 == 0:
            self._update_hourly_stats()
    
    def _cleanup_old_records(self):
        """Remove records older than 7 days from database."""
        cutoff_time = datetime.now().timestamp() - (7 * 24 * 3600)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM cav_memory WHERE timestamp < ?", (cutoff_time,))
        conn.commit()
        conn.close()
    
    def _update_hourly_stats(self):
        """
        Update hourly statistics using EWMA (Exponential Weighted Moving Average).
        
        Computes for each hour [0-23]:
        - cav_mu[hour]: Mean CAV (EWMA)
        - cav_var[hour]: Variance CAV (EWMA)
        - state_probs[hour]: State frequency distribution
        """
        # Filter to last 24 hours
        cutoff_time = datetime.now().timestamp() - self.window_seconds
        recent_records = [
            r for r in self.buffer
            if r['timestamp'] >= cutoff_time
        ]
        
        # Group by hour
        hourly_data: Dict[int, List[Dict]] = {h: [] for h in range(24)}
        for record in recent_records:
            hour = record['local_hour']
            hourly_data[hour].append(record)
        
        # Compute EWMA statistics for each hour
        alpha = 0.3  # EWMA smoothing factor
        
        for hour in range(24):
            hour_records = hourly_data[hour]
            
            if not hour_records:
                # No data for this hour, use default or previous stats
                if hour in self.hourly_stats:
                    # Keep previous stats
                    continue
                else:
                    # Initialize with defaults
                    self.hourly_stats[hour] = {
                        'mu': 5000.0,  # Mid-range CAV
                        'var': 2500000.0,  # Variance estimate
                        'state_probs': {
                            'overload': 0.25,
                            'balanced': 0.25,
                            'focus': 0.25,
                            'restorative': 0.25
                        }
                    }
                continue
            
            # Extract CAV values
            cav_values = [r['cav'] for r in hour_records]
            
            # Compute mean (EWMA)
            current_mean = statistics.mean(cav_values)
            if hour in self.hourly_stats:
                prev_mean = self.hourly_stats[hour]['mu']
                new_mean = alpha * current_mean + (1 - alpha) * prev_mean
            else:
                new_mean = current_mean
            
            # Compute variance (EWMA)
            current_var = statistics.variance(cav_values) if len(cav_values) > 1 else 2500000.0
            if hour in self.hourly_stats:
                prev_var = self.hourly_stats[hour]['var']
                new_var = alpha * current_var + (1 - alpha) * prev_var
            else:
                new_var = current_var
            
            # Compute state frequency distribution
            state_counts = {'overload': 0, 'balanced': 0, 'focus': 0, 'restorative': 0}
            for r in hour_records:
                state = r['state']
                if state in state_counts:
                    state_counts[state] += 1
            
            total = len(hour_records)
            state_probs = {
                state: count / total if total > 0 else 0.25
                for state, count in state_counts.items()
            }
            
            # Update stats
            self.hourly_stats[hour] = {
                'mu': new_mean,
                'var': max(new_var, 100000.0),  # Minimum variance floor
                'state_probs': state_probs
            }
    
    def compute_adaptive(
        self,
        cav_smooth: int,
        state: str,
        aqi: int,
        local_hour: int
    ) -> Dict[str, float]:
        """
        Compute adaptive adjustments based on current context and historical patterns.
        
        Args:
            cav_smooth: Current smoothed CAV score
            state: Current state
            aqi: Current Air Quality Index
            local_hour: Current local hour [0-23]
        
        Returns:
            Dictionary with z_cav, sensitivity, and env_weight_adj
        """
        # Get baseline statistics for current hour
        if local_hour not in self.hourly_stats:
            # No data for this hour, use default baseline
            baseline_mu = 5000.0
            baseline_std = math.sqrt(2500000.0)  # ~1581
        else:
            stats = self.hourly_stats[local_hour]
            baseline_mu = stats['mu']
            baseline_std = math.sqrt(stats['var'])
        
        # Compute z-score
        if baseline_std > 0:
            z_cav = (cav_smooth - baseline_mu) / baseline_std
        else:
            z_cav = 0.0
        
        # Compute sensitivity adjustment
        # If |z_cav| > 1.5, increase sensitivity (faster state changes)
        if abs(z_cav) > 1.5:
            sensitivity = 1.0 + min(abs(z_cav) - 1.5, 0.5) * 0.5  # Max 1.25x
        else:
            sensitivity = 1.0
        
        # Compute environment weight adjustment
        # If AQI consistently bad, lower environment weighting
        # Check recent AQI patterns
        cutoff_time = datetime.now().timestamp() - (6 * 3600)  # Last 6 hours
        recent_records = [
            r for r in self.buffer
            if r['timestamp'] >= cutoff_time
        ]
        
        if recent_records:
            # Count bad AQI (AQI > 100)
            bad_aqi_count = sum(1 for r in recent_records if r['aqi'] > 100)
            bad_aqi_ratio = bad_aqi_count / len(recent_records)
            
            # If >50% of recent readings are bad AQI, reduce env weight
            if bad_aqi_ratio > 0.5:
                env_weight_adj = 0.8  # Reduce by 20%
            elif bad_aqi_ratio > 0.3:
                env_weight_adj = 0.9  # Reduce by 10%
            else:
                env_weight_adj = 1.0  # No adjustment
        else:
            # No recent data, use current AQI
            if aqi > 100:
                env_weight_adj = 0.9
            else:
                env_weight_adj = 1.0
        
        return {
            'z_cav': round(z_cav, 2),
            'sensitivity': round(sensitivity, 2),
            'env_weight_adj': round(env_weight_adj, 2)
        }
    
    def get_summary(self) -> Dict:
        """
        Get 24-hour summary of memory statistics.
        
        Returns:
            Dictionary with baselines, state probabilities, and current averages
        """
        # Get recent records (last 24h)
        cutoff_time = datetime.now().timestamp() - self.window_seconds
        recent_records = [
            r for r in self.buffer
            if r['timestamp'] >= cutoff_time
        ]
        
        if not recent_records:
            return {
                'total_records': 0,
                'window_hours': self.window_hours,
                'hourly_stats': {},
                'overall_stats': {
                    'cav_mean': 0.0,
                    'cav_std': 0.0,
                    'state_distribution': {}
                }
            }
        
        # Overall statistics
        cav_values = [r['cav'] for r in recent_records]
        cav_mean = statistics.mean(cav_values)
        cav_std = statistics.stdev(cav_values) if len(cav_values) > 1 else 0.0
        
        # State distribution
        state_counts = {'overload': 0, 'balanced': 0, 'focus': 0, 'restorative': 0}
        for r in recent_records:
            state = r['state']
            if state in state_counts:
                state_counts[state] += 1
        
        total = len(recent_records)
        state_distribution = {
            state: round(count / total, 3) if total > 0 else 0.0
            for state, count in state_counts.items()
        }
        
        # Format hourly stats
        hourly_stats_formatted = {}
        for hour in range(24):
            if hour in self.hourly_stats:
                stats = self.hourly_stats[hour]
                hourly_stats_formatted[hour] = {
                    'cav_mean': round(stats['mu'], 1),
                    'cav_std': round(math.sqrt(stats['var']), 1),
                    'state_probs': {
                        k: round(v, 3) for k, v in stats['state_probs'].items()
                    }
                }
        
        return {
            'total_records': len(recent_records),
            'window_hours': self.window_hours,
            'hourly_stats': hourly_stats_formatted,
            'overall_stats': {
                'cav_mean': round(cav_mean, 1),
                'cav_std': round(cav_std, 1),
                'state_distribution': state_distribution
            }
        }
    
    def clear(self):
        """Clear all memory (buffer and database)."""
        self.buffer.clear()
        self.hourly_stats.clear()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cav_memory")
        conn.commit()
        conn.close()

