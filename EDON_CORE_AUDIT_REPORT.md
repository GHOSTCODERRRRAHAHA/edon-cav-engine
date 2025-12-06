# EDON CORE AUDIT REPORT

## 1. ENVIRONMENT & PROFILE LAYER: ✅ PASS

**Findings:**
- `--profile` argument is parsed in `run_eval.py:788` via `argparse`
- `get_stress_profile(args.profile)` called at `run_eval.py:824`
- `EnvironmentRandomizer` constructed with stress profile at `run_eval.py:825`
- Profile values are logged at `run_eval.py:826` showing `push_prob` and `noise_std`
- Profiles have distinct configurations:
  - `light_stress`: push_prob=0.08, noise_std=0.01
  - `medium_stress`: push_prob=0.12, noise_std=0.02
  - `high_stress`: push_prob=0.18, noise_std=0.03
  - `hell_stress`: push_prob=0.65, noise_std=0.04
- `EnvironmentRandomizer` uses profile values at `evaluation/randomization.py:29-36` for `push_probability` and `sensor_noise_std`
- Profile is applied in `apply_step_randomization()` at `evaluation/randomization.py:125` using `self.push_probability`

**Location:**
- `run_eval.py:788, 824-826`
- `evaluation/stress_profiles.py:92-97, 100-104`
- `evaluation/randomization.py:12-47, 102-133`

---

## 2. BASELINE vs EDON CONTROLLER SWITCHING: ✅ PASS

**Findings:**
- `--mode` argument parsed at `run_eval.py:784` with choices `["baseline", "edon"]`
- `use_edon` boolean set at `run_eval.py:829`: `use_edon = (args.mode == "edon")`
- Controller selection logic at `run_eval.py:857-867`:
  - If `use_edon`: creates wrapper calling `edon_controller()` with `edon_gain`
  - Else: assigns `baseline_controller` directly
- `HumanoidRunner` receives `use_edon` flag at `run_eval.py:878`
- Runner branches on `use_edon` at `evaluation/humanoid_runner.py:165` to conditionally get EDON state
- Controller called at `evaluation/humanoid_runner.py:169`: `action = self.controller(obs, edon_state)`
- In EDON mode, `edon_controller()` is called; in baseline mode, `baseline_controller()` is called
- Baseline controller is NOT used in EDON mode (EDON mode calls `edon_controller()` which internally calls `baseline_controller()` only to compute baseline action, then applies EDON regulation)

**Location:**
- `run_eval.py:784, 829, 857-867, 878`
- `evaluation/humanoid_runner.py:165, 169`
- `run_eval.py:731-761` (edon_controller function)

---

## 3. EDON CONTROLLER EXISTS & IS STATEFUL: ✅ PASS

**Findings:**
- `EDONController` class exists at `run_eval.py:115-427`
- `EDONAdaptiveState` dataclass at `run_eval.py:92-112` with persistent fields:
  - `last_gain`, `instability_score`, `phase`, `smoothed_tilt`, `smoothed_vel_norm`, etc.
- Controller instantiated once per evaluation run at `run_eval.py:903`: `_edon_controller_instance = EDONController(base_edon_gain=args.edon_gain)`
- Global instance stored at `run_eval.py:431`: `_edon_controller_instance: Optional[EDONController] = None`
- Controller retrieved/reused in `apply_edon_regulation()` at `run_eval.py:496-505`:
  - Checks if instance exists, creates if None
  - Updates base_gain if changed
  - Reuses same instance across all steps
- Controller persists across episode: `reset_episode()` at `run_eval.py:420-427` resets phase/episode tracking but preserves smoothed signals
- State persists across steps: `adaptive_state` is member of controller instance, not recreated

**Location:**
- `run_eval.py:92-112` (EDONAdaptiveState)
- `run_eval.py:115-427` (EDONController class)
- `run_eval.py:431` (global instance)
- `run_eval.py:496-505` (instance retrieval)
- `run_eval.py:903` (initialization)
- `run_eval.py:420-427` (episode reset)

---

## 4. INSTABILITY SCORING PIPELINE: ✅ PASS

**Findings:**
- `compute_instability_score()` method exists at `run_eval.py:201-260`
- Consumes real runtime signals:
  - Tilt: `roll`, `pitch` from `obs` → `tilt_magnitude = sqrt(roll^2 + pitch^2)`
  - Velocities: `roll_velocity`, `pitch_velocity` from `obs` → `vel_norm = sqrt(roll_velocity^2 + pitch_velocity^2)`
  - EDON signals: `p_chaos`, `p_stress` from `edon_state_raw`
  - Risk EMA: `risk_ema` parameter
  - Tilt zone: `tilt_zone` parameter
- Formula at `run_eval.py:248-254`:
  ```python
  instability = (
      0.35 * tilt_normalized +
      0.25 * vel_normalized +
      0.20 * p_chaos +
      0.15 * p_stress +
      0.05 * risk_ema
  )
  ```
- Boost applied for prefall/fail zones: `instability *= 1.15` at `run_eval.py:258`
- Returns bounded scalar `[0, 1]` via `max(0.0, min(1.0, instability))` at `run_eval.py:260`
- Called every step at `run_eval.py:562-564`: `instability_score = controller.compute_instability_score(obs, edon_state_raw, tilt_zone, risk_ema)`
- Score is tracked in history: `controller.adaptive_state.instability_history.append(instability_score)` at `run_eval.py:573`

**Location:**
- `run_eval.py:201-260` (compute_instability_score method)
- `run_eval.py:562-564` (call site)
- `run_eval.py:573` (history tracking)

---

## 5. PHASE STATE MACHINE (STABLE / WARNING / RECOVERY): ✅ PASS

**Findings:**
- Phase stored as string in `EDONAdaptiveState.phase` at `run_eval.py:102`: `phase: str = "stable"`
- Three phases: `"stable"`, `"warning"`, `"recovery"`
- Phase transitions in `update_phase()` at `run_eval.py:262-303`
- Hysteresis thresholds defined at `run_eval.py:170-173`:
  - `T_WARNING_ON = 0.5` (enter warning)
  - `T_WARNING_OFF = 0.3` (exit warning)
  - `T_RECOVERY_ON = 0.85` (enter recovery)
  - `T_RECOVERY_OFF = 0.2` (exit recovery)
- Transition logic at `run_eval.py:280-296`:
  - `stable → warning`: if `instability_score > T_WARNING_ON`
  - `warning → recovery`: if `instability_score > T_RECOVERY_ON`
  - `recovery → warning`: if `instability_score < T_RECOVERY_OFF`
  - `warning → stable`: if `instability_score < T_WARNING_OFF`
- Phase updated per step at `run_eval.py:570`: `phase = controller.update_phase(controller.adaptive_state.instability_score)`
- Phase stored in state: `self.adaptive_state.phase = new_phase` at `run_eval.py:300`
- Phase counts tracked: `self.adaptive_state.phase_counts[new_phase] += 1` at `run_eval.py:301`
- Phase logged in episode summary at `run_eval.py:407-412`

**Location:**
- `run_eval.py:102` (phase field)
- `run_eval.py:170-173` (thresholds)
- `run_eval.py:262-303` (update_phase method)
- `run_eval.py:570` (update call)
- `run_eval.py:407-412` (logging)

---

## 6. ADAPTIVE GAIN COMPUTATION: ✅ PASS

**Findings:**
- `--edon-gain` parsed at `run_eval.py:794` and stored as `args.edon_gain`
- Base gain stored in controller: `self.base_edon_gain = base_edon_gain` at `run_eval.py:165`
- `compute_adaptive_gain()` method at `run_eval.py:305-339`
- Formula at `run_eval.py:314-329`:
  - Phase multipliers: `GAIN_STABLE=0.4`, `GAIN_WARNING=0.8`, `GAIN_RECOVERY=1.0`
  - Raw gain: `adaptive_gain = self.base_edon_gain * phase_gain`
  - Recovery cap: `adaptive_gain = min(adaptive_gain, GAIN_RECOVERY_MAX)` where `GAIN_RECOVERY_MAX=1.0`
  - Smoothing: `adaptive_gain = ALPHA_GAIN * adaptive_gain + (1 - ALPHA_GAIN) * last_gain` with `ALPHA_GAIN=0.2`
- Last gain stored: `self.adaptive_state.last_gain = adaptive_gain` at `run_eval.py:337`
- Gain history tracked: `controller.adaptive_state.gain_history.append(adaptive_gain)` at `run_eval.py:622`
- Called every step at `run_eval.py:621`: `adaptive_gain = controller.compute_adaptive_gain()`
- Base gain influence preserved: base_gain is multiplied by phase_gain, not replaced

**Location:**
- `run_eval.py:794` (CLI argument)
- `run_eval.py:165` (base gain storage)
- `run_eval.py:181-186` (phase multipliers)
- `run_eval.py:305-339` (compute_adaptive_gain method)
- `run_eval.py:621-622` (call and history)

---

## 7. PREFALL vs SAFE TORQUE BLENDING: ✅ PASS

**Findings:**
- `prefall_torque` computed at `run_eval.py:646-656`:
  - Base correction from PD controller
  - Scaled by `prefall_gain` (0.18-0.65 range based on `fall_risk`)
  - Only active if `internal_zone in ("prefall", "fail")`
- `safe_torque` computed at `run_eval.py:659-667`:
  - Direct roll/pitch correction: `safe_torque[0] = -0.15 * roll * SAFE_GAIN`
  - Only active if `catastrophic_risk > 0.75`
- Phase-dependent weights from `get_torque_weights()` at `run_eval.py:341-355`:
  - Stable: `w_prefall=0.2`, `w_safe=0.8`
  - Warning: `w_prefall=0.5`, `w_safe=0.5`
  - Recovery: `w_prefall=0.5`, `w_safe=0.5`
- Weights retrieved at `run_eval.py:674`: `w_prefall, w_safe = controller.get_torque_weights()`
- Blending at `run_eval.py:677`: `combined_torque = w_prefall * prefall_torque + w_safe * safe_torque`
- Blended torque multiplied by adaptive_gain at `run_eval.py:685`: `edon_contribution = adaptive_gain * combined_torque`
- Added to baseline at `run_eval.py:686`: `final_action = baseline_action + edon_contribution`
- Both torques can be active simultaneously (blending, not exclusive)

**Location:**
- `run_eval.py:646-656` (prefall_torque)
- `run_eval.py:659-667` (safe_torque)
- `run_eval.py:341-355` (get_torque_weights method)
- `run_eval.py:674, 677, 685-686` (blending and final action)

---

## 8. FINAL ACTION PATH (NO OVERRIDES): ⚠️ PARTIAL

**Findings:**
- Final action computed at `run_eval.py:686`: `final_action = baseline_action + edon_contribution`
- Action clamped at `run_eval.py:694-698` via `clamp_action_relative_to_baseline()`:
  - Limits EDON action to max 1.2x baseline magnitude (`EDON_MAX_CORRECTION_RATIO = 1.2`)
  - This is a safety clamp, not an override
- Action clamped again at `run_eval.py:701-705` (duplicate clamp - non-critical)
- Final clip at `run_eval.py:708`: `action = np.clip(action, -1.0, 1.0)` (hardware limits)
- Action returned from `apply_edon_regulation()` at `run_eval.py:728`
- Action returned from `edon_controller()` at `run_eval.py:761`: `return final_action[0]`
- Action sent to environment at `evaluation/humanoid_runner.py:172`: `obs, reward, done, info = self.env.step(action)`

**⚠️ WARNING:**
- `clamp_action_relative_to_baseline()` at `run_eval.py:457-479` can significantly reduce EDON contribution if it exceeds 20% of baseline magnitude
- This is intentional safety but may limit EDON effectiveness in high-gain scenarios
- Duplicate clamp at lines 694 and 701 is redundant but harmless

**Location:**
- `run_eval.py:686` (final_action computation)
- `run_eval.py:694-698, 701-705` (clamping)
- `run_eval.py:708` (hardware clip)
- `run_eval.py:728, 761` (return)
- `evaluation/humanoid_runner.py:172` (env.step)

---

## 9. METRICS & TEST HARNESS INTEGRITY: ✅ PASS

**Findings:**
- `run_eval.py` computes metrics for both baseline and EDON modes
- Metrics aggregated at `run_eval.py:921`: `run_metrics = aggregate_run_metrics(episode_metrics_list, mode=args.mode)`
- JSON output at `run_eval.py:927-941` includes:
  - `mode`, `profile`, `episodes`, `seed`, `edon_gain`
  - `run_metrics`: `interventions_per_episode`, `freeze_events_per_episode`, `stability_avg`, `avg_episode_length`, `success_rate`
  - `episodes`: array of episode metrics
- `fast_test.py` runs baseline first at `fast_test.py:181`: `run_baseline(...)`
- `fast_test.py` runs EDON second at `fast_test.py:186`: `run_edon(...)`
- Improvement computation at `fast_test.py:80-112`:
  - `interv_improvement = ((baseline_int - edon_int) / baseline_int) * 100.0`
  - `stab_improvement = ((baseline_stab - edon_stab) / baseline_stab) * 100.0`
  - `average = (interv_improvement + stab_improvement) / 2.0`
- No normalization or rescaling of EDON-only metrics
- Results saved to JSON at `fast_test.py:131-161` with baseline, EDON, and improvements

**Location:**
- `run_eval.py:921, 927-941` (metrics aggregation and JSON)
- `fast_test.py:15-39` (run_baseline)
- `fast_test.py:42-67` (run_edon)
- `fast_test.py:80-112` (compute_improvements)
- `fast_test.py:131-161` (save_results)

---

## 10. LOGGING & VISIBILITY: ✅ PASS

**Findings:**
- Configuration logged on first episode at `run_eval.py:388-393`:
  - `base_gain`, `thresholds` (warning_on/off, recovery_on/off)
- Per-episode summary logged at `run_eval.py:407-412`:
  - `phase_counts` (stable/warning/recovery step counts)
  - `avg_instability` (average instability score)
  - `avg_gain` (average adaptive gain)
- Phase logged in debug info at `run_eval.py:719`: `'phase': phase`
- Adaptive gain logged on first call at `run_eval.py:690`: `[EDON-ADAPTIVE] base_gain={edon_gain:.3f} adaptive_gain={adaptive_gain:.3f} phase={phase}`
- Controller gain logged at `run_eval.py:748`: `[EDON-CONTROLLER] gain={edon_gain}`
- Episode summary called at `run_eval.py:916`: `_edon_controller_instance.log_episode_summary(episode_id)`
- Runtime visibility: Phase, gain, and instability are visible in logs

**Location:**
- `run_eval.py:388-393` (config logging)
- `run_eval.py:407-412` (episode summary)
- `run_eval.py:690` (first-call logging)
- `run_eval.py:719` (debug info)
- `run_eval.py:916` (summary call)

---

## OVERALL CORE STATUS: ⚠️ PARTIAL

**Summary:**
- 9 out of 10 components: ✅ PASS
- 1 component: ⚠️ PARTIAL (Final Action Path - safety clamping may limit effectiveness)

---

## CRITICAL BLOCKERS (if any):

- **None identified.** All core components exist, are wired, and are actively used.

---

## NON-CRITICAL WEAKNESSES:

1. **Action Clamping May Limit EDON Effectiveness:**
   - `clamp_action_relative_to_baseline()` limits EDON contribution to 120% of baseline magnitude
   - This safety feature may prevent EDON from applying strong corrections when needed
   - Location: `run_eval.py:457-479, 694-698, 701-705`
   - Impact: EDON corrections may be reduced in high-instability scenarios

2. **Duplicate Action Clamping:**
   - Action is clamped twice (lines 694 and 701) - redundant but harmless
   - Location: `run_eval.py:694-698, 701-705`

3. **Recovery Phase Very Hard to Enter:**
   - `T_RECOVERY_ON = 0.85` means recovery phase is rarely triggered (requires 85% instability)
   - This may be intentional (conservative design) but limits adaptive behavior
   - Location: `run_eval.py:172`

4. **Baseline Controller Called in EDON Mode:**
   - `edon_controller()` calls `baseline_controller()` internally to compute baseline action
   - This is correct (EDON adds to baseline), but baseline is always computed even if EDON should override
   - Location: `run_eval.py:754`

---

## RECOMMENDATIONS:

1. Consider making `EDON_MAX_CORRECTION_RATIO` phase-dependent (higher in recovery)
2. Remove duplicate clamp at line 701
3. Monitor recovery phase activation rate - if never triggered, consider lowering `T_RECOVERY_ON`
4. Add runtime logging for clamp events to understand when EDON corrections are being limited

---

**Report Generated:** Based on codebase analysis of `run_eval.py`, `fast_test.py`, `evaluation/humanoid_runner.py`, `evaluation/randomization.py`, and `evaluation/stress_profiles.py`

