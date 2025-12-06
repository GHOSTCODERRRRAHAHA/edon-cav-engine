"""
Example integration showing how to adapt the evaluation system
to your specific humanoid simulator.

This file shows the key integration points you need to modify.
"""

# ============================================================================
# STEP 1: Replace make_humanoid_env() in run_eval.py
# ============================================================================

def make_humanoid_env_example(seed=None):
    """
    Example: Replace this function in run_eval.py with your environment.
    
    Common patterns:
    """
    # Option 1: Gym environment
    # import gym
    # env = gym.make("Humanoid-v3")
    # if seed is not None:
    #     env.seed(seed)
    # return env
    
    # Option 2: MuJoCo environment
    # from mujoco_py import MjSim, load_model_from_path
    # model = load_model_from_path("humanoid.xml")
    # env = MjSim(model)
    # return env
    
    # Option 3: PyBullet environment
    # import pybullet_envs
    # env = gym.make("HumanoidBulletEnv-v0")
    # return env
    
    # Option 4: Custom environment
    # from my_robot_sim import HumanoidSimulator
    # env = HumanoidSimulator(seed=seed)
    # return env
    
    pass


# ============================================================================
# STEP 2: Update _extract_stability_metrics() in humanoid_runner.py
# ============================================================================

def extract_stability_metrics_example(obs):
    """
    Example: Adapt this to your observation space.
    
    Your observation might be:
    - A dictionary: obs["torso_roll"], obs["com_x"]
    - A numpy array: obs[0], obs[1], obs[2]
    - A custom object: obs.torso_roll, obs.center_of_mass
    """
    
    # Example 1: Dictionary observation
    if isinstance(obs, dict):
        roll = obs.get("torso_roll", obs.get("roll", 0.0))
        pitch = obs.get("torso_pitch", obs.get("pitch", 0.0))
        com_x = obs.get("com_x", obs.get("center_of_mass_x", 0.0))
        com_y = obs.get("com_y", obs.get("center_of_mass_y", 0.0))
        com = (com_x**2 + com_y**2)**0.5
        return roll, pitch, com
    
    # Example 2: Numpy array observation
    elif isinstance(obs, (list, tuple)) or hasattr(obs, '__len__'):
        # Assume fixed indices (adjust to your observation space)
        roll = obs[0]   # Adjust index
        pitch = obs[1]  # Adjust index
        com_x = obs[2]  # Adjust index
        com_y = obs[3]  # Adjust index
        com = (com_x**2 + com_y**2)**0.5
        return roll, pitch, com
    
    # Example 3: Custom object
    else:
        roll = getattr(obs, 'torso_roll', getattr(obs, 'roll', 0.0))
        pitch = getattr(obs, 'torso_pitch', getattr(obs, 'pitch', 0.0))
        com = getattr(obs, 'com', 0.0)
        return roll, pitch, com


# ============================================================================
# STEP 3: Update controller functions in run_eval.py
# ============================================================================

def baseline_controller_example(obs, edon_state=None):
    """
    Example: Replace with your actual baseline control policy.
    
    This should return an action that your environment expects.
    """
    # Example 1: Simple policy
    # return np.random.uniform(-1, 1, size=env.action_space.shape)
    
    # Example 2: PID controller
    # from my_controllers import PIDController
    # controller = PIDController()
    # return controller.compute(obs)
    
    # Example 3: RL policy
    # import torch
    # policy = torch.load("baseline_policy.pt")
    # action = policy(obs)
    # return action
    
    pass


def edon_controller_example(obs, edon_state=None):
    """
    Example: Controller that uses EDON state to modulate behavior.
    """
    # Get base action from baseline controller
    base_action = baseline_controller_example(obs, None)
    
    if edon_state is None:
        return base_action
    
    # Extract EDON influences
    influences = edon_state.get("influences", {})
    speed_scale = influences.get("speed_scale", 1.0)
    torque_scale = influences.get("torque_scale", 1.0)
    emergency = influences.get("emergency_flag", False)
    
    # Apply influences
    if emergency:
        # Emergency stop
        return np.zeros_like(base_action)
    else:
        # Scale action
        return base_action * speed_scale


# ============================================================================
# STEP 4: Update _build_sensor_window() in humanoid_runner.py
# ============================================================================

class SensorBuffer:
    """
    Example: Buffer for accumulating sensor readings for EDON.
    
    EDON needs 240 samples per signal (4 seconds @ 60Hz).
    In real-time, you'd buffer actual sensor readings.
    """
    
    def __init__(self, window_size=240):
        self.window_size = window_size
        self.eda_buffer = []
        self.bvp_buffer = []
        self.acc_x_buffer = []
        self.acc_y_buffer = []
        self.acc_z_buffer = []
    
    def add_reading(self, eda, bvp, acc_x, acc_y, acc_z):
        """Add a single sensor reading."""
        self.eda_buffer.append(eda)
        self.bvp_buffer.append(bvp)
        self.acc_x_buffer.append(acc_x)
        self.acc_y_buffer.append(acc_y)
        self.acc_z_buffer.append(acc_z)
        
        # Keep only last window_size samples
        if len(self.eda_buffer) > self.window_size:
            self.eda_buffer.pop(0)
            self.bvp_buffer.pop(0)
            self.acc_x_buffer.pop(0)
            self.acc_y_buffer.pop(0)
            self.acc_z_buffer.pop(0)
    
    def build_window(self, env_data):
        """Build EDON window from buffered sensor data."""
        return {
            "physio": {
                "EDA": self.eda_buffer[-self.window_size:] if len(self.eda_buffer) >= self.window_size else [0.1] * self.window_size,
                "BVP": self.bvp_buffer[-self.window_size:] if len(self.bvp_buffer) >= self.window_size else [0.5] * self.window_size
            },
            "motion": {
                "ACC_x": self.acc_x_buffer[-self.window_size:] if len(self.acc_x_buffer) >= self.window_size else [0.0] * self.window_size,
                "ACC_y": self.acc_y_buffer[-self.window_size:] if len(self.acc_y_buffer) >= self.window_size else [0.0] * self.window_size,
                "ACC_z": self.acc_z_buffer[-self.window_size:] if len(self.acc_z_buffer) >= self.window_size else [1.0] * self.window_size
            },
            "env": {
                "temp_c": env_data.get("temp_c", 22.0),
                "humidity": env_data.get("humidity", 45.0),
                "aqi": env_data.get("aqi", 20)
            },
            "task": {
                "id": env_data.get("task_id", "simulation"),
                "complexity": env_data.get("task_complexity", 0.5)
            }
        }


# ============================================================================
# STEP 5: Update _detect_intervention() if needed
# ============================================================================

def detect_intervention_example(obs, roll, pitch):
    """
    Example: Customize intervention detection for your robot.
    
    You might want to check:
    - Joint limits
    - Torque limits
    - Velocity limits
    - Contact forces
    - Battery level
    - etc.
    """
    from evaluation.config import config
    
    # Check for fall
    if abs(roll) > config.FALL_THRESHOLD_ROLL or abs(pitch) > config.FALL_THRESHOLD_PITCH:
        return True
    
    # Check for torque violations (if available in obs)
    if "joint_torques" in obs:
        max_torque = np.max(np.abs(obs["joint_torques"]))
        if max_torque > config.SAFETY_TORQUE_LIMIT:
            return True
    
    # Add your custom checks here
    # if obs.get("battery_level", 1.0) < 0.1:
    #     return True
    
    return False

