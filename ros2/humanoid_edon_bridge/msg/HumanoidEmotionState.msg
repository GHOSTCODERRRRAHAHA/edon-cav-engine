# High-level emotional/context state for a humanoid powered by EDON.

# EDON's primary mode:
#  - "overload"
#  - "balanced"
#  - "focus"
#  - "restorative"
#  - "unknown"
string edon_mode

# 0.0–1.0 model confidence in this state
float32 state_confidence

# Probability of stress (from EDON parts.p_stress)
float32 stress_probability

# Behavior profiles the robot should adopt.
# These are semantic labels; each stack maps them to low-level commands.
string locomotion_profile   # e.g. "slow", "normal", "fast"
string voice_profile        # e.g. "soft", "neutral", "crisp"
string gaze_profile         # e.g. "soft_follow", "direct"
string environment_profile  # e.g. "dim_lights", "normal_lights"

# Optional EDON component scores (0–1)
float32 bio_score
float32 env_score
float32 circadian_score

# Timestamp when EDON computed this state
builtin_interfaces/Time stamp

