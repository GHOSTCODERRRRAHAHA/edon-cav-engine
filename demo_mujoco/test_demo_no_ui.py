"""Test the demo without UI to verify it runs for full duration."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sim.env import HumanoidEnv
from controllers.baseline_controller import BaselineController
from controllers.edon_layer import EdonLayer
from disturbances.generator import DisturbanceGenerator
from metrics.tracker import MetricsTracker
import numpy as np
import time

# Import stress profile from evaluation module
try:
    from evaluation.stress_profiles import get_stress_profile, HIGH_STRESS
    STRESS_PROFILE_AVAILABLE = True
except ImportError:
    # Fallback if evaluation module not available
    STRESS_PROFILE_AVAILABLE = False
    print("⚠️  Warning: evaluation.stress_profiles not available, using defaults")


def test_comparison():
    """Test the comparison without UI."""
    print("="*60)
    print("Testing EDON MuJoCo Demo (No UI)")
    print("="*60)
    
    seed = 42
    episode_duration = 10.0  # Shorter episodes (1000 steps) - more focused like original
    dt = 0.01
    max_steps = int(episode_duration / dt)  # 1000 steps instead of 3000
    
    print(f"Episode duration: {episode_duration}s ({max_steps} steps)")
    print(f"Seed: {seed}")
    print()
    
    # Get stress profile (HIGH_STRESS matches training/eval environment)
    if STRESS_PROFILE_AVAILABLE:
        stress_profile = HIGH_STRESS
        print(f"Using stress profile: {stress_profile.name}")
        print(f"  - Sensor noise: {stress_profile.sensor_noise_std}")
        print(f"  - Actuator delay: {stress_profile.actuator_delay_steps} steps")
        print(f"  - Friction range: {stress_profile.friction_min}-{stress_profile.friction_max}")
        print(f"  - Fatigue: {stress_profile.fatigue_enabled} ({stress_profile.fatigue_degradation*100:.0f}% degradation)")
        print(f"  - Floor incline: ±{max(abs(stress_profile.floor_incline_range[0]), abs(stress_profile.floor_incline_range[1]))*180/np.pi:.1f}°")
    else:
        # Default high stress parameters
        stress_profile = type('StressProfile', (), {
            'sensor_noise_std': 0.03,
            'actuator_delay_steps': (2, 4),
            'friction_min': 0.2,
            'friction_max': 1.5,
            'fatigue_enabled': True,
            'fatigue_degradation': 0.10,
            'floor_incline_range': (-0.15, 0.15),
            'height_variation_range': (-0.05, 0.05)
        })()
    
    # Create environments with stress profile parameters
    print("Creating environments...")
    env_baseline = HumanoidEnv(
        dt=dt, 
        render=False,
        sensor_noise_std=stress_profile.sensor_noise_std,
        actuator_delay_steps=stress_profile.actuator_delay_steps,
        friction_min=stress_profile.friction_min,
        friction_max=stress_profile.friction_max,
        fatigue_enabled=stress_profile.fatigue_enabled,
        fatigue_degradation=stress_profile.fatigue_degradation,
        floor_incline_range=stress_profile.floor_incline_range,
        height_variation_range=stress_profile.height_variation_range
    )
    env_edon = HumanoidEnv(
        dt=dt, 
        render=False,
        sensor_noise_std=stress_profile.sensor_noise_std,
        actuator_delay_steps=stress_profile.actuator_delay_steps,
        friction_min=stress_profile.friction_min,
        friction_max=stress_profile.friction_max,
        fatigue_enabled=stress_profile.fatigue_enabled,
        fatigue_degradation=stress_profile.fatigue_degradation,
        floor_incline_range=stress_profile.floor_incline_range,
        height_variation_range=stress_profile.height_variation_range
    )
    print("✓ Environments created with realism features")
    
    # Create controllers
    print("Creating controllers...")
    baseline_controller = BaselineController()
    edon_layer = EdonLayer(edon_base_url="http://localhost:8000", enabled=True)
    print("✓ Controllers created")
    
    # Check EDON status
    if edon_layer.enabled and edon_layer.client is not None:
        print("✓ EDON layer is ENABLED and connected")
        try:
            # Test call
            test_state = {
                "roll": 0.0,
                "pitch": 0.0,
                "roll_velocity": 0.0,
                "pitch_velocity": 0.0,
                "com_x": 0.0,
                "com_y": 0.0
            }
            test_result = edon_layer.client.robot_stability(test_state)
            print(f"✓ EDON test call successful: strategy={test_result.get('strategy_name', 'N/A')}")
        except Exception as e:
            print(f"⚠️  EDON test call failed: {e}")
    else:
        print("⚠️  EDON layer is DISABLED or not connected!")
        print("   Make sure EDON server is running: python -m app.main")
    
    # Create disturbance generator
    print("Generating disturbance scripts...")
    dist_gen = DisturbanceGenerator(seed=seed)
    
    # Baseline: HIGH STRESS
    script_baseline = dist_gen.generate_script(
        duration=episode_duration,
        dt=dt,
        push_probability=0.3,  # High stress
        terrain_bumps=10,
        load_shifts=6,
        latency_jitter_enabled=True
    )
    # Amplify forces
    for event in script_baseline:
        if event.get("type") == "push":
            force = event.get("force", [0, 0, 0])
            event["force"] = [f * 1.5 for f in force]
    
    # EDON: SAME HIGH STRESS (to show EDON handles it better)
    script_edon = dist_gen.generate_script(
        duration=episode_duration,
        dt=dt,
        push_probability=0.3,  # Same high stress as baseline
        terrain_bumps=10,
        load_shifts=6,
        latency_jitter_enabled=True
    )
    # Same amplified forces
    for event in script_edon:
        if event.get("type") == "push":
            force = event.get("force", [0, 0, 0])
            event["force"] = [f * 1.5 for f in force]
    
    print(f"  Baseline (HIGH STRESS): {len(script_baseline)} events")
    print(f"  EDON (SAME HIGH STRESS): {len(script_baseline)} events (using same script)")
    print("  Note: Both face IDENTICAL disturbances to show EDON's advantage")
    print()
    
    # Create metrics trackers
    metrics_baseline = MetricsTracker()
    metrics_edon = MetricsTracker()
    
    # ===== RUN BASELINE =====
    print("="*60)
    print("Running BASELINE episode (HIGH STRESS)...")
    print("="*60)
    
    obs, info = env_baseline.reset(seed=seed, disturbance_script=script_baseline)
    metrics_baseline.reset()
    
    baseline_steps = 0
    start_time = time.time()
    
    for step in range(max_steps):
        baseline_steps = step + 1
        
        # Get action
        action = baseline_controller.step(obs)
        
        # Step environment
        obs, done, info = env_baseline.step(action)
        
        # Update metrics
        metrics_baseline.update(obs, info, dt=dt)
        
        # Print progress
        if baseline_steps % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  [Baseline] Step {baseline_steps}/{max_steps} ({baseline_steps*dt:.1f}s/{episode_duration}s) - "
                  f"Roll={obs.get('roll', 0):.3f}, Pitch={obs.get('pitch', 0):.3f}, "
                  f"Height={obs.get('torso_height', 0):.3f}, Falls={metrics_baseline.falls}")
        
        # Small delay
        time.sleep(dt)
        
        # Don't break on done - continue for full duration
        if done and baseline_steps < max_steps:
            # Reset if catastrophic
            height = obs.get('torso_height', 1.0)
            if height > 5.0 or height < 0.1:
                print(f"  [Baseline] Catastrophic failure at step {baseline_steps}, resetting...")
                obs, info = env_baseline.reset(seed=seed + baseline_steps, disturbance_script=script_baseline)
    
    baseline_time = time.time() - start_time
    baseline_metrics = metrics_baseline.get_metrics()
    
    print(f"\n[Baseline] Complete: {baseline_steps}/{max_steps} steps in {baseline_time:.1f}s")
    print(f"  Falls: {baseline_metrics['falls']}")
    print(f"  Freezes: {baseline_metrics['freezes']}")
    print(f"  Interventions: {baseline_metrics['interventions']}")
    print(f"  Stability Score: {baseline_metrics['stability_score']:.2f}")
    print()
    
    # ===== RUN EDON =====
    print("="*60)
    print("Running EDON episode (SAME HIGH STRESS)...")
    print("="*60)
    
    # Use SAME seed and SAME script for fair comparison
    obs, info = env_edon.reset(seed=seed, disturbance_script=script_baseline)  # Same as baseline!
    metrics_edon.reset()
    edon_layer.reset()
    
    edon_steps = 0
    start_time = time.time()
    
    for step in range(max_steps):
        edon_steps = step + 1
        
        # Get baseline action
        baseline_action = baseline_controller.step(obs)
        
        # Apply EDON
        action, edon_info = edon_layer.step(obs, baseline_action)
        
        # Step environment
        obs, done, info = env_edon.step(action)
        
        # Update metrics
        metrics_edon.update(obs, info, dt=dt)
        
        # Print progress
        if edon_steps % 500 == 0:
            elapsed = time.time() - start_time
            risk = edon_info.get('intervention_risk', 0)
            enabled = edon_info.get('enabled', False)
            strategy = edon_info.get('strategy_name', 'N/A')
            gain_scale = edon_info.get('gain_scale', 1.0)
            compliance = edon_info.get('compliance', 1.0)
            baseline_mag = np.linalg.norm(baseline_action)
            action_mag = np.linalg.norm(action)
            print(f"  [EDON] Step {edon_steps}/{max_steps} ({edon_steps*dt:.1f}s/{episode_duration}s) - "
                  f"Roll={obs.get('roll', 0):.3f}, Pitch={obs.get('pitch', 0):.3f}, "
                  f"Height={obs.get('torso_height', 0):.3f}, Risk={risk:.3f}, Falls={metrics_edon.falls}")
            lateral_compliance = edon_info.get('lateral_compliance', 1.0)
            step_height_bias = edon_info.get('step_height_bias', 0.0)
            print(f"         EDON: enabled={enabled}, strategy={strategy}, gain_scale={gain_scale:.2f}")
            print(f"         Modulations: lateral_compliance={lateral_compliance:.2f}, step_height_bias={step_height_bias:.3f}")
            print(f"         Action: baseline_mag={baseline_mag:.2f}, final_mag={action_mag:.2f}, diff={action_mag-baseline_mag:.2f}")
            # Show normalized action magnitudes to verify normalization is working
            baseline_norm = np.linalg.norm(baseline_action / 20.0)
            final_norm = np.linalg.norm(action / 20.0)
            print(f"         Normalized: baseline_norm={baseline_norm:.3f}, final_norm={final_norm:.3f}, change={final_norm-baseline_norm:.3f}")
            if not enabled:
                error = edon_info.get('error', '')
                if error:
                    print(f"         ERROR: {error}")
            elif edon_steps == 500:  # First detailed check
                print(f"         First EDON call details:")
                print(f"           - Strategy ID: {edon_info.get('strategy_id', -1)}")
                print(f"           - Latency: {edon_info.get('latency_ms', 0):.2f}ms")
                print(f"           - Baseline action sample: {baseline_action[:3]}")
                print(f"           - Final action sample: {action[:3]}")
        
        # Small delay
        time.sleep(dt)
        
        # Don't break on done - continue for full duration
        if done and edon_steps < max_steps:
            # Reset if catastrophic
            height = obs.get('torso_height', 1.0)
            if height > 5.0 or height < 0.1:
                print(f"  [EDON] Catastrophic failure at step {edon_steps}, resetting...")
                obs, info = env_edon.reset(seed=seed + edon_steps, disturbance_script=script_baseline)  # Same script
    
    edon_time = time.time() - start_time
    edon_metrics = metrics_edon.get_metrics()
    
    print(f"\n[EDON] Complete: {edon_steps}/{max_steps} steps in {edon_time:.1f}s")
    print(f"  Falls: {edon_metrics['falls']}")
    print(f"  Freezes: {edon_metrics['freezes']}")
    print(f"  Interventions: {edon_metrics['interventions']}")
    print(f"  Stability Score: {edon_metrics['stability_score']:.2f}")
    print()
    
    # ===== COMPARISON =====
    print("="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Baseline (HIGH STRESS):")
    print(f"  Steps: {baseline_steps}/{max_steps} ({baseline_steps/max_steps*100:.1f}%)")
    print(f"  Falls: {baseline_metrics['falls']}")
    print(f"  Freezes: {baseline_metrics['freezes']}")
    print(f"  Interventions: {baseline_metrics['interventions']}")
    print(f"  Stability: {baseline_metrics['stability_score']:.2f}")
    print()
    print(f"EDON (SAME HIGH STRESS):")
    print(f"  Steps: {edon_steps}/{max_steps} ({edon_steps/max_steps*100:.1f}%)")
    print(f"  Falls: {edon_metrics['falls']}")
    print(f"  Freezes: {edon_metrics['freezes']}")
    print(f"  Interventions: {edon_metrics['interventions']}")
    print(f"  Stability: {edon_metrics['stability_score']:.2f}")
    print()
    
    # Calculate improvement
    fall_reduction = ((baseline_metrics['falls'] - edon_metrics['falls']) / max(baseline_metrics['falls'], 1)) * 100
    stability_improvement = edon_metrics['stability_score'] - baseline_metrics['stability_score']
    
    print("="*60)
    print("IMPROVEMENT ANALYSIS")
    print("="*60)
    print(f"Fall Reduction: {fall_reduction:.1f}%")
    print(f"Stability Improvement: {stability_improvement:+.2f}")
    if fall_reduction > 0 or stability_improvement > 0:
        print("✅ EDON shows improvement!")
    elif fall_reduction == 0 and stability_improvement == 0:
        print("⚠️  Similar performance - check EDON connection")
    else:
        print("⚠️  EDON performed worse - investigate")
    print()
    
    if baseline_steps < max_steps:
        print(f"⚠️  WARNING: Baseline ended early! ({baseline_steps}/{max_steps})")
    if edon_steps < max_steps:
        print(f"⚠️  WARNING: EDON ended early! ({edon_steps}/{max_steps})")
    
    if baseline_steps == max_steps and edon_steps == max_steps:
        print("✅ Both episodes completed for full duration!")
    
    print("="*60)


if __name__ == "__main__":
    print("Starting test...")
    try:
        test_comparison()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

