#!/bin/bash
# Commit and push EDON v8 work

cd "$(dirname "$0")"

echo "Staging EDON v8 files..."

# Core v8 implementation files
git add training/edon_v8_policy.py
git add training/train_edon_v8_strategy.py
git add env/edon_humanoid_env_v8.py
git add metrics/edon_v8_metrics.py
git add eval_v8_memory_features.py
git add eval_v8_multiple_seeds.py

# Documentation
git add EDON_V8_FULL_ARCHITECTURE.md
git add INTERVENTION_REDUCTION_SUMMARY.md
git add VERSION_COMPARISON.md
git add EDON_V8_VS_CORE_ARCHITECTURE.md
git add EDON_CORE_IMPORTANCE_ANALYSIS.md

# Results (if not in .gitignore)
git add results/edon_v8_memory_features.json 2>/dev/null || true
git add results/edon_v8_generalization_test.json 2>/dev/null || true

echo "Committing..."
git commit -m "EDON v8: Temporal memory and predictive modulation for intervention minimization

- Added temporal memory: 8-frame stacked observations (248 dims)
- Added early-warning features: rolling variance, oscillation energy, near-fail density
- Integrated fail-risk prediction model for proactive intervention prevention
- Implemented layered control: learned strategy + modulations on stable baseline
- Achieved 97.5% intervention reduction (1.00/episode vs 40.30 baseline)
- Maintained stability within ±5% constraint (0.0215 vs 0.0208 baseline)
- Validated generalization across seeds (0, 42, 100, 200) with consistent performance

Key files:
- training/edon_v8_policy.py: Policy network with temporal memory support
- env/edon_humanoid_env_v8.py: Environment wrapper with memory buffers
- metrics/edon_v8_metrics.py: Fixed intervention detection
- eval_v8_memory_features.py: Evaluation script
- eval_v8_multiple_seeds.py: Generalization testing

Results:
- Interventions: 1.00/episode (97.5% reduction from baseline)
- Stability: 0.0215 (within ±5% of baseline)
- Generalization: Consistent across all test seeds"

echo "Pushing to remote..."
git push

echo "Done!"

