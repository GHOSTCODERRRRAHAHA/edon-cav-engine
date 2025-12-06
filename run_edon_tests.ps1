# Comprehensive EDON test suite for all profiles and gains
# Run this script to test EDON controller across multiple scenarios

$profiles = @("normal_stress", "high_stress", "hell_stress")
$gains = @(0.60, 0.75, 0.90, 1.00)
$episodes = 30

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "EDON COMPREHENSIVE TEST SUITE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Profiles: $($profiles -join ', ')" -ForegroundColor Yellow
Write-Host "Gains: $($gains -join ', ')" -ForegroundColor Yellow
Write-Host "Episodes per test: $episodes" -ForegroundColor Yellow
Write-Host "Total tests: $($profiles.Count * $gains.Count)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$testCount = 0
$totalTests = $profiles.Count * $gains.Count

foreach ($profile in $profiles) {
    foreach ($gain in $gains) {
        $testCount++
        $tag = "{0:000}" -f [int]($gain * 100)
        $output = "results/edon_${profile}_v44_g${tag}.json"
        
        Write-Host "[$testCount/$totalTests] " -NoNewline -ForegroundColor Green
        Write-Host "EDON $profile gain=$gain (tag $tag)" -ForegroundColor Cyan
        
        python run_eval.py `
            --mode edon `
            --episodes $episodes `
            --profile $profile `
            --edon-gain $gain `
            --edon-controller-version v3 `
            --output $output
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Test failed!" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "✓ Completed" -ForegroundColor Green
        Write-Host ""
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ALL TESTS COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Results saved to: results/" -ForegroundColor Yellow
Write-Host "Next: Run plot_results.py to visualize comparisons" -ForegroundColor Yellow

