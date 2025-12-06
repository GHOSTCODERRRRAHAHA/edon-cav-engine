@echo off
REM Train v8 Strategy Policy on D Drive
REM This script trains the policy and saves models to D drive

echo ========================================================================
echo EDON v8 Strategy Policy Training - D Drive
echo ========================================================================
echo.

REM Create D drive directory if it doesn't exist
if not exist "D:\edon_models" mkdir "D:\edon_models"
if not exist "D:\edon_models\logs" mkdir "D:\edon_models\logs"

echo Training will save models to: D:\edon_models\
echo.

REM Train the policy
python training/train_edon_v8_strategy.py ^
  --episodes 300 ^
  --profile high_stress ^
  --seed 0 ^
  --lr 5e-4 ^
  --gamma 0.995 ^
  --update-epochs 10 ^
  --output-dir "D:\edon_models" ^
  --model-name edon_v8_strategy_v1_trained ^
  --fail-risk-model models/edon_fail_risk_v1_fixed.pt ^
  --max-steps 1000

echo.
echo ========================================================================
echo Training complete!
echo Model saved to: D:\edon_models\edon_v8_strategy_v1_trained.pt
echo ========================================================================
pause

