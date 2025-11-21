# Build release package for EDON v1.0.0 (PowerShell)

$ErrorActionPreference = "Stop"
$VERSION = "v1.0.0"
$RELEASE_DIR = "release\$VERSION"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Building EDON Release $VERSION" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Create release directory
Write-Host "`nCreating release directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $RELEASE_DIR | Out-Null

# Build Python wheel
Write-Host "`nBuilding Python wheel..." -ForegroundColor Yellow
Push-Location sdk\python

# Check if build tools are available
try {
    python -c "import build" 2>$null
} catch {
    Write-Host "Installing build tools..." -ForegroundColor Yellow
    pip install build wheel
}

# Build wheel
python -m build --wheel

# Move wheel to release directory
$wheel = Get-ChildItem dist\*.whl | Select-Object -First 1
Move-Item $wheel.FullName "$RELEASE_DIR\" -Force
Write-Host "Python wheel built: $($wheel.Name)" -ForegroundColor Green

Pop-Location

# Build C++ SDK
Write-Host "`nBuilding C++ SDK..." -ForegroundColor Yellow
Push-Location sdk\cpp

# Clean previous build
if (Test-Path "build") {
    Remove-Item -Recurse -Force build
}

# Create build directory
New-Item -ItemType Directory -Force -Path build | Out-Null
Push-Location build

# Configure and build
cmake ..
cmake --build . --config Release

Pop-Location
Pop-Location

# Create C++ SDK package
$SDK_PACKAGE_DIR = "$RELEASE_DIR\edon-sdk-cpp-$VERSION"
New-Item -ItemType Directory -Force -Path $SDK_PACKAGE_DIR | Out-Null

# Copy headers
Copy-Item -Recurse -Force sdk\cpp\include "$SDK_PACKAGE_DIR\"

# Copy libraries
$libDir = "$SDK_PACKAGE_DIR\lib"
New-Item -ItemType Directory -Force -Path $libDir | Out-Null

if (Test-Path "sdk\cpp\build\Release\edon_sdk.lib") {
    Copy-Item "sdk\cpp\build\Release\edon_sdk.lib" "$libDir\" -Force
    Write-Host "Static library copied" -ForegroundColor Green
} elseif (Test-Path "sdk\cpp\build\libedon_sdk.a") {
    Copy-Item "sdk\cpp\build\libedon_sdk.a" "$libDir\" -Force
    Write-Host "Static library copied" -ForegroundColor Green
} elseif (Test-Path "sdk\cpp\build\libedon_sdk.so") {
    Copy-Item "sdk\cpp\build\libedon_sdk.so" "$libDir\" -Force
    Write-Host "Shared library copied" -ForegroundColor Green
}

# Copy example binary
$binDir = "$SDK_PACKAGE_DIR\bin"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null

if (Test-Path "sdk\cpp\build\Release\edon_robot_example.exe") {
    Copy-Item "sdk\cpp\build\Release\edon_robot_example.exe" "$binDir\" -Force
    Write-Host "Example binary copied" -ForegroundColor Green
} elseif (Test-Path "sdk\cpp\build\bin\edon_robot_example") {
    Copy-Item "sdk\cpp\build\bin\edon_robot_example" "$binDir\" -Force
    Write-Host "Example binary copied" -ForegroundColor Green
} elseif (Test-Path "sdk\cpp\build\edon_robot_example") {
    Copy-Item "sdk\cpp\build\edon_robot_example" "$binDir\" -Force
    Write-Host "Example binary copied" -ForegroundColor Green
}

# Copy CMake files
Copy-Item "sdk\cpp\CMakeLists.txt" "$SDK_PACKAGE_DIR\" -Force
if (Test-Path "sdk\cpp\examples") {
    Copy-Item -Recurse -Force "sdk\cpp\examples" "$SDK_PACKAGE_DIR\"
}

# Create tarball (using 7zip or tar if available)
Push-Location $RELEASE_DIR
if (Get-Command tar -ErrorAction SilentlyContinue) {
    tar -czf "edon-sdk-cpp-$VERSION.tar.gz" "edon-sdk-cpp-$VERSION"
    Write-Host "C++ SDK package created: edon-sdk-cpp-$VERSION.tar.gz" -ForegroundColor Green
} elseif (Get-Command 7z -ErrorAction SilentlyContinue) {
    7z a -ttar "edon-sdk-cpp-$VERSION.tar" "edon-sdk-cpp-$VERSION"
    7z a -tgzip "edon-sdk-cpp-$VERSION.tar.gz" "edon-sdk-cpp-$VERSION.tar"
    Remove-Item "edon-sdk-cpp-$VERSION.tar"
    Write-Host "C++ SDK package created: edon-sdk-cpp-$VERSION.tar.gz" -ForegroundColor Green
} else {
    Write-Host "Warning: tar or 7z not found. Creating zip instead..." -ForegroundColor Yellow
    Compress-Archive -Path "edon-sdk-cpp-$VERSION" -DestinationPath "edon-sdk-cpp-$VERSION.zip" -Force
    Write-Host "C++ SDK package created: edon-sdk-cpp-$VERSION.zip" -ForegroundColor Green
}
Remove-Item -Recurse -Force "edon-sdk-cpp-$VERSION"
Pop-Location

# Build Docker image
Write-Host "`nBuilding Docker image..." -ForegroundColor Yellow
docker build -t "edon-server:$VERSION" .

# Save Docker image
Write-Host "Saving Docker image..." -ForegroundColor Yellow
docker save "edon-server:$VERSION" | Out-File -FilePath "$RELEASE_DIR\edon-server-$VERSION.docker" -Encoding binary

# Also create compressed version
Write-Host "Creating compressed Docker image..." -ForegroundColor Yellow
docker save "edon-server:$VERSION" | gzip > "$RELEASE_DIR\edon-server-$VERSION.docker.tar.gz"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Release build complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Release files in: $RELEASE_DIR\" -ForegroundColor Yellow
Get-ChildItem $RELEASE_DIR | Format-Table Name, Length -AutoSize
Write-Host ""
Write-Host "To load Docker image:" -ForegroundColor Yellow
Write-Host "  docker load < $RELEASE_DIR\edon-server-$VERSION.docker" -ForegroundColor White
Write-Host ""
Write-Host "To install Python wheel:" -ForegroundColor Yellow
Write-Host "  pip install $RELEASE_DIR\*.whl" -ForegroundColor White
Write-Host ""

