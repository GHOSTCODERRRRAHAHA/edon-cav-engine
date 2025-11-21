# EDON C++ SDK

This C++ SDK provides a lightweight client for the EDON CAV Engine gRPC or REST interfaces, suitable for robotics and embedded OEM integrations.

## Requirements

- CMake 3.16+
- A C++17-compatible compiler (GCC, Clang, or MSVC)
- gRPC and Protobuf development libraries (if using the gRPC transport)

## Build (generic Linux / WSL2)

From the repo root:

```bash
cd sdk/cpp
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
```

This will produce:
- A library (e.g. `libedon_sdk.a` or `edon_sdk.lib`)
- Example binaries if provided (e.g., `edon_robot_example`)

## Install / package

You can install or stage artifacts via CMake:

```bash
cmake --install . --prefix ./install
```

This will create a structure like:

```
sdk/cpp/build/install/
├── include/        # Public headers
└── lib/            # Compiled library
```

You can then zip this directory for OEM delivery:

```bash
cd sdk/cpp/build
zip -r ../../release/edon-cpp-sdk-v1.0.1.zip install
```

## Integration

Typical OEM integration flow:

1. Link your robotics / embedded application against `libedon_sdk` (or the equivalent library produced by this project).

2. Configure the EDON endpoint:
   - REST: base URL of the EDON Docker deployment (e.g. `http://edon:8000`)
   - gRPC: host/port (e.g. `edon:50051`)

3. Call the CAV helper with a 240-sample window of physiological + environmental signals.

4. Use the resulting CAV state to modulate speed, torque, or safety limits in your control stack.

## Example Usage

```cpp
#include "edon/edon.hpp"

// Create client
edon::EdonClient client("localhost", 50051);

// Create sensor window
edon::SensorWindow window;
window.eda = std::vector<float>(240, 0.1f);
window.temp = std::vector<float>(240, 36.5f);
window.bvp = std::vector<float>(240, 0.5f);
window.acc_x = std::vector<float>(240, 0.0f);
window.acc_y = std::vector<float>(240, 0.0f);
window.acc_z = std::vector<float>(240, 1.0f);
window.temp_c = 22.0f;
window.humidity = 50.0f;
window.aqi = 35;
window.local_hour = 14;

// Compute CAV
edon::CAVResponse response = client.computeCAV(window);

// Use state to modulate robot behavior
if (response.state == "overload") {
    // Reduce speed/torque, increase safety
} else if (response.state == "focus") {
    // Increase performance
}
```

## Building on Windows

```powershell
cd sdk\cpp
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

## Building on macOS

```bash
cd sdk/cpp
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```
