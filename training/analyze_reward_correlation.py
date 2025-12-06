"""
Analyze correlation between step_reward and EDON Score.

Runs a fixed policy (baseline or heuristic) and collects:
- Per-episode total reward (sum of step_reward)
- Per-episode EDON Score
- Per-episode interventions, stability, length

Then computes correlation and suggests reward adjustments.
"""

import sys
from pathlib import Path
import numpy as np
import json
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.edon_humanoid_env import EdonHumanoidEnv
from training.edon_score import step_reward, compute_episode_score
from run_eval import baseline_controller


def collect_episode_data(env: EdonHumanoidEnv, num_episodes: int = 30) -> List[Dict[str, Any]]:
    """Collect episode data using baseline controller."""
    episodes = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        
        episode_rewards = []
        episode_interventions = 0
        episode_stability_scores = []
        episode_length = 0
        
        while not done and step < 1000:
            try:
                # Use baseline controller
                baseline_action = baseline_controller(obs, edon_state=None)
                baseline_action = np.array(baseline_action)
                
                # Store baseline for delta computation (needed for reward)
                env.last_baseline_action = baseline_action
                
                # Step environment (will compute EDON reward internally)
                next_obs, reward, done, info = env.step(baseline_action)
            except Exception as e:
                print(f"Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
                break
            
            episode_rewards.append(reward)
            episode_length += 1
            
            # Track interventions
            if info.get("intervention", False) or info.get("fallen", False):
                episode_interventions += 1
            
            # Track stability (from observation)
            roll = abs(next_obs.get("roll", 0.0))
            pitch = abs(next_obs.get("pitch", 0.0))
            stability_score = (roll + pitch) / 2.0  # Rough stability metric
            episode_stability_scores.append(stability_score)
            
            obs = next_obs
            step += 1
        
        # Compute episode metrics
        total_reward = sum(episode_rewards)
        avg_stability = np.mean(episode_stability_scores) if episode_stability_scores else 0.0
        
        episode_summary = {
            "interventions": episode_interventions,
            "stability_score": avg_stability,
            "episode_length": episode_length
        }
        edon_score = compute_episode_score(episode_summary)
        
        episodes.append({
            "episode": episode,
            "total_reward": total_reward,
            "edon_score": edon_score,
            "interventions": episode_interventions,
            "stability": avg_stability,
            "length": episode_length,
            "reward_per_step": total_reward / episode_length if episode_length > 0 else 0.0
        })
        
        if (episode + 1) % 10 == 0:
            print(f"Collected {episode + 1}/{num_episodes} episodes...")
    
    return episodes


def analyze_correlation(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze correlation between reward and EDON Score."""
    total_rewards = np.array([e["total_reward"] for e in episodes])
    edon_scores = np.array([e["edon_score"] for e in episodes])
    interventions = np.array([e["interventions"] for e in episodes])
    stabilities = np.array([e["stability"] for e in episodes])
    lengths = np.array([e["length"] for e in episodes])
    
    # Compute correlations
    reward_score_corr = np.corrcoef(total_rewards, edon_scores)[0, 1]
    reward_int_corr = np.corrcoef(total_rewards, interventions)[0, 1]
    reward_stab_corr = np.corrcoef(total_rewards, stabilities)[0, 1]
    reward_len_corr = np.corrcoef(total_rewards, lengths)[0, 1]
    
    # Compute statistics
    reward_mean = np.mean(total_rewards)
    reward_std = np.std(total_rewards)
    score_mean = np.mean(edon_scores)
    score_std = np.std(edon_scores)
    
    # Analyze what drives EDON Score
    # EDON Score = 0.4 * intervention_score + 0.4 * stability_score + 0.2 * length_bonus
    # intervention_score = max(0, 100 - interventions * 2)
    # stability_score = 100 * (1 - min(1, stability * 10))
    # length_bonus = min(20, length / 50)
    
    intervention_scores = np.array([max(0.0, 100.0 - (i * 2.0)) for i in interventions])
    stability_scores = np.array([100.0 * (1.0 - min(1.0, s * 10.0)) for s in stabilities])
    length_bonuses = np.array([min(20.0, l / 50.0) for l in lengths])
    
    # Compute what EDON Score components correlate with reward
    try:
        reward_int_score_corr = float(np.corrcoef(total_rewards, intervention_scores)[0, 1])
        if np.isnan(reward_int_score_corr) or np.isinf(reward_int_score_corr):
            reward_int_score_corr = 0.0
    except:
        reward_int_score_corr = 0.0
    
    try:
        reward_stab_score_corr = float(np.corrcoef(total_rewards, stability_scores)[0, 1])
        if np.isnan(reward_stab_score_corr) or np.isinf(reward_stab_score_corr):
            reward_stab_score_corr = 0.0
    except:
        reward_stab_score_corr = 0.0
    
    try:
        reward_len_bonus_corr = float(np.corrcoef(total_rewards, length_bonuses)[0, 1])
        if np.isnan(reward_len_bonus_corr) or np.isinf(reward_len_bonus_corr):
            reward_len_bonus_corr = 0.0
    except:
        reward_len_bonus_corr = 0.0
    
    return {
        "reward_score_correlation": float(reward_score_corr),
        "reward_interventions_correlation": float(reward_int_corr),
        "reward_stability_correlation": float(reward_stab_corr),
        "reward_length_correlation": float(reward_len_corr),
        "reward_intervention_score_correlation": float(reward_int_score_corr),
        "reward_stability_score_correlation": float(reward_stab_score_corr),
        "reward_length_bonus_correlation": float(reward_len_bonus_corr),
        "reward_stats": {
            "mean": float(reward_mean),
            "std": float(reward_std),
            "min": float(np.min(total_rewards)),
            "max": float(np.max(total_rewards))
        },
        "edon_score_stats": {
            "mean": float(score_mean),
            "std": float(score_std),
            "min": float(np.min(edon_scores)),
            "max": float(np.max(edon_scores))
        },
        "intervention_stats": {
            "mean": float(np.mean(interventions)),
            "std": float(np.std(interventions)),
            "min": int(np.min(interventions)),
            "max": int(np.max(interventions))
        },
        "stability_stats": {
            "mean": float(np.mean(stabilities)),
            "std": float(np.std(stabilities)),
            "min": float(np.min(stabilities)),
            "max": float(np.max(stabilities))
        }
    }


def suggest_reward_adjustments(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest reward function adjustments based on correlation analysis."""
    suggestions = {}
    
    reward_score_corr = analysis["reward_score_correlation"]
    reward_int_corr = analysis["reward_interventions_correlation"]
    reward_stab_corr = analysis["reward_stability_correlation"]
    
    # Target: reward should correlate POSITIVELY with EDON Score
    # EDON Score increases when:
    # - Interventions decrease (negative correlation with interventions)
    # - Stability improves (negative correlation with stability value, positive with stability_score)
    
    print("\n" + "="*70)
    print("REWARD ADJUSTMENT SUGGESTIONS")
    print("="*70)
    
    # Check reward-EDON Score correlation
    if reward_score_corr < 0.5:
        print(f"\n[ISSUE] Reward-EDON Score correlation is {reward_score_corr:.3f} (target: >0.7)")
        print("   Reward function is not well aligned with EDON Score!")
        suggestions["increase_alignment"] = True
    else:
        print(f"\n[OK] Reward-EDON Score correlation is {reward_score_corr:.3f}")
        suggestions["increase_alignment"] = False
    
    # Check intervention correlation
    # Reward should be NEGATIVELY correlated with interventions (more interventions = lower reward)
    if reward_int_corr > -0.3:
        print(f"\n[ISSUE] Reward-Interventions correlation is {reward_int_corr:.3f} (target: <-0.5)")
        print("   Intervention penalty may be too weak!")
        suggestions["increase_intervention_penalty"] = True
        suggestions["suggested_intervention_penalty"] = -30.0  # Increase from -20.0
    else:
        print(f"\n[OK] Reward-Interventions correlation is {reward_int_corr:.3f}")
        suggestions["increase_intervention_penalty"] = False
    
    # Check stability correlation
    # Reward should be NEGATIVELY correlated with stability value (higher stability value = worse)
    # But POSITIVELY correlated with stability_score (higher stability_score = better)
    if reward_stab_corr > -0.3:
        print(f"\n[ISSUE] Reward-Stability correlation is {reward_stab_corr:.3f} (target: <-0.4)")
        print("   Stability penalty may be too weak!")
        suggestions["increase_stability_penalty"] = True
        suggestions["suggested_tilt_penalty"] = -10.0  # Increase from -8.0
        suggestions["suggested_velocity_penalty"] = -4.0  # Increase from -3.0
    else:
        print(f"\n[OK] Reward-Stability correlation is {reward_stab_corr:.3f}")
        suggestions["increase_stability_penalty"] = False
    
    # Check reward scale
    reward_mean = analysis["reward_stats"]["mean"]
    if reward_mean < -2000:
        print(f"\n[ISSUE] Reward mean is {reward_mean:.1f} (target: -200 to -800)")
        print("   Rewards are too negative, may need scaling!")
        suggestions["scale_rewards"] = True
        suggestions["suggested_scale_factor"] = abs(-500 / reward_mean)  # Scale to ~-500
    else:
        print(f"\n[OK] Reward mean is {reward_mean:.1f}")
        suggestions["scale_rewards"] = False
    
    return suggestions


def main():
    """Main analysis."""
    try:
        print("="*70)
        print("Reward <-> EDON Score Correlation Analysis")
        print("="*70)
        print()
        
        # Create environment
        print("Creating environment...")
        env = EdonHumanoidEnv(seed=42, profile="high_stress")
        print("Environment created successfully")
        
        # Collect episode data
        print("Collecting episode data using baseline controller...")
        episodes = collect_episode_data(env, num_episodes=30)
        print(f"Collected {len(episodes)} episodes")
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # Analyze correlation
    print("\nAnalyzing correlations...")
    analysis = analyze_correlation(episodes)
    
    # Print results
    print("\n" + "="*70)
    print("CORRELATION RESULTS")
    print("="*70)
    print(f"\nReward <-> EDON Score: {analysis['reward_score_correlation']:.3f}")
    print(f"Reward <-> Interventions: {analysis['reward_interventions_correlation']:.3f}")
    print(f"Reward <-> Stability: {analysis['reward_stability_correlation']:.3f}")
    print(f"Reward <-> Length: {analysis['reward_length_correlation']:.3f}")
    print(f"\nReward <-> Intervention Score: {analysis['reward_intervention_score_correlation']:.3f}")
    print(f"Reward <-> Stability Score: {analysis['reward_stability_score_correlation']:.3f}")
    print(f"Reward <-> Length Bonus: {analysis['reward_length_bonus_correlation']:.3f}")
    
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"\nReward:")
    print(f"  Mean: {analysis['reward_stats']['mean']:.2f}")
    print(f"  Std: {analysis['reward_stats']['std']:.2f}")
    print(f"  Range: [{analysis['reward_stats']['min']:.2f}, {analysis['reward_stats']['max']:.2f}]")
    
    print(f"\nEDON Score:")
    print(f"  Mean: {analysis['edon_score_stats']['mean']:.2f}")
    print(f"  Std: {analysis['edon_score_stats']['std']:.2f}")
    print(f"  Range: [{analysis['edon_score_stats']['min']:.2f}, {analysis['edon_score_stats']['max']:.2f}]")
    
    print(f"\nInterventions:")
    print(f"  Mean: {analysis['intervention_stats']['mean']:.2f}")
    print(f"  Range: [{analysis['intervention_stats']['min']}, {analysis['intervention_stats']['max']}]")
    
    print(f"\nStability:")
    print(f"  Mean: {analysis['stability_stats']['mean']:.4f}")
    print(f"  Range: [{analysis['stability_stats']['min']:.4f}, {analysis['stability_stats']['max']:.4f}]")
    
    # Get suggestions
    suggestions = suggest_reward_adjustments(analysis)
    
    # Save results
    output = {
        "episodes": episodes,
        "analysis": analysis,
        "suggestions": suggestions
    }
    
    output_path = Path("training/reward_correlation_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    
    return analysis, suggestions


if __name__ == "__main__":
    main()

