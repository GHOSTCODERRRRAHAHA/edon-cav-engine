# Test script for EDON Core Robot Stability API

$baseUrl = "http://localhost:8002"
$endpoint = "$baseUrl/oem/robot/stability"

# Test robot state
$body = @{
    robot_state = @{
        roll = 0.05
        pitch = 0.02
        roll_velocity = 0.1
        pitch_velocity = 0.05
        com_x = 0.0
        com_y = 0.0
    }
} | ConvertTo-Json -Depth 10

Write-Host "Testing Robot Stability API..."
Write-Host "Endpoint: $endpoint"
Write-Host "Request body:"
Write-Host $body
Write-Host ""

try {
    $response = Invoke-RestMethod -Uri $endpoint -Method Post -Body $body -ContentType "application/json"
    
    Write-Host "✅ Success!"
    Write-Host "Response:"
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "❌ Error:"
    Write-Host $_.Exception.Message
    if ($_.ErrorDetails) {
        Write-Host "Details:"
        Write-Host $_.ErrorDetails.Message
    }
}

