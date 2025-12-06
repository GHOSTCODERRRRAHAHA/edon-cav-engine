"""Generate deterministic disturbance scripts for testing."""

import numpy as np
from typing import List, Dict, Any, Optional
import json


class DisturbanceGenerator:
    """
    Generate deterministic disturbance scripts for reproducible testing.
    
    Disturbances include:
    - Impulse pushes (lateral + frontal)
    - Uneven terrain (heightfield bumps)
    - Dynamic load shifts
    - Latency jitter
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed for reproducibility."""
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
    
    def generate_script(
        self,
        duration: float = 30.0,
        dt: float = 0.01,
        push_probability: float = 0.1,
        terrain_bumps: int = 5,
        load_shifts: int = 3,
        latency_jitter_enabled: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate a disturbance script.
        
        Args:
            duration: Episode duration in seconds
            dt: Timestep
            push_probability: Probability of push per second
            terrain_bumps: Number of terrain height changes
            load_shifts: Number of load position shifts
            latency_jitter_enabled: Whether to include latency jitter
        
        Returns:
            List of disturbance events, sorted by time
        """
        script = []
        n_steps = int(duration / dt)
        
        # Generate impulse pushes
        push_interval = 1.0 / push_probability  # Expected time between pushes
        current_time = push_interval
        
        while current_time < duration:
            # Random direction (lateral or frontal)
            direction = self.rng.choice(["lateral", "frontal", "diagonal"])
            
            if direction == "lateral":
                # Lateral push (sideways)
                force = [
                    self.rng.uniform(-300, 300),  # X force (increased from 200)
                    self.rng.uniform(-75, 75),    # Y force (increased from 50)
                    0.0                           # Z force
                ]
            elif direction == "frontal":
                # Frontal push (forward/back)
                force = [
                    self.rng.uniform(-75, 75),    # X force (increased from 50)
                    self.rng.uniform(-300, 300),  # Y force (increased from 200)
                    0.0                           # Z force
                ]
            else:  # diagonal
                force = [
                    self.rng.uniform(-225, 225),  # Increased from 150
                    self.rng.uniform(-225, 225),  # Increased from 150
                    0.0
                ]
            
            # Apply point on torso
            point = [
                self.rng.uniform(-0.1, 0.1),
                self.rng.uniform(-0.1, 0.1),
                1.0  # Torso height
            ]
            
            script.append({
                "type": "push",
                "time": current_time,
                "force": force,
                "point": point
            })
            
            # Next push time
            current_time += push_interval * self.rng.uniform(0.5, 1.5)
        
        # Generate terrain bumps
        for i in range(terrain_bumps):
            time = self.rng.uniform(2.0, duration - 2.0)
            height = self.rng.uniform(-0.05, 0.05)  # Small bumps
            
            script.append({
                "type": "terrain",
                "time": time,
                "height": height
            })
        
        # Generate load shifts
        for i in range(load_shifts):
            time = self.rng.uniform(3.0, duration - 3.0)
            position = [
                self.rng.uniform(-0.2, 0.2),
                self.rng.uniform(-0.2, 0.2),
                self.rng.uniform(0.0, 0.1)
            ]
            
            script.append({
                "type": "load_shift",
                "time": time,
                "position": position
            })
        
        # Generate latency jitter periods
        if latency_jitter_enabled:
            for i in range(3):
                time = self.rng.uniform(5.0, duration - 5.0)
                jitter = self.rng.uniform(0.01, 0.05)  # 10-50ms jitter
                duration_jitter = self.rng.uniform(1.0, 3.0)  # Jitter period
                
                script.append({
                    "type": "latency_jitter",
                    "time": time,
                    "jitter": jitter,
                    "duration": duration_jitter
                })
        
        # Sort by time
        script.sort(key=lambda x: x["time"])
        
        return script
    
    def save_script(self, script: List[Dict[str, Any]], filepath: str):
        """Save disturbance script to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(script, f, indent=2)
    
    def load_script(self, filepath: str) -> List[Dict[str, Any]]:
        """Load disturbance script from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)

