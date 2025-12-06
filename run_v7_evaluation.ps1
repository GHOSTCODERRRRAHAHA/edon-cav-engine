# Run baseline and v7 evaluations for comparison

Write-Host "Running baseline evaluation..."
python run_eval.py `
  --mode baseline `
  --profile high_stress `
  --episodes 30 `
  --seed 42 `
  --output results/baseline_ep200.json `
  --edon-score

Write-Host ""
Write-Host "Running v7 evaluation..."
python run_eval.py `
  --mode edon `
  --profile high_stress `
  --episodes 30 `
  --seed 42 `
  --output results/edon_v7_ep200.json `
  --edon-gain 1.0 `
  --edon-arch v7_learned `
  --edon-score

Write-Host ""
Write-Host "Running comparison..."
python training/compare_v7_ep200.py

