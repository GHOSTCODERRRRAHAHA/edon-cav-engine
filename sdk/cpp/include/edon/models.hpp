/**
 * @file models.hpp
 * @brief Data models for EDON SDK
 */

#pragma once

#include <string>
#include <vector>
#include <map>

namespace edon {

/**
 * @brief Sensor window data structure
 */
struct SensorWindow {
    std::vector<float> eda;      // EDA signal (240 samples)
    std::vector<float> temp;     // Temperature signal (240 samples)
    std::vector<float> bvp;      // BVP signal (240 samples)
    std::vector<float> acc_x;    // ACC X (240 samples)
    std::vector<float> acc_y;    // ACC Y (240 samples)
    std::vector<float> acc_z;     // ACC Z (240 samples)
    
    float temp_c = 22.0f;         // Ambient temperature (°C)
    float humidity = 50.0f;       // Relative humidity (%)
    int aqi = 50;                 // Air Quality Index
    int local_hour = 12;          // Local hour [0-23]
};

/**
 * @brief Component scores breakdown
 */
struct ComponentScores {
    float bio = 0.0f;
    float env = 0.0f;
    float circadian = 0.0f;
    float p_stress = 0.0f;
};

/**
 * @brief Robot control scales
 */
struct ControlScales {
    float speed = 1.0f;
    float torque = 1.0f;
    float safety = 0.7f;
};

/**
 * @brief CAV computation response
 */
struct CAVResponse {
    int cav_raw = 0;              // Raw CAV score [0-10000]
    int cav_smooth = 0;           // Smoothed CAV score [0-10000]
    std::string state;             // State: overload, balanced, focus, restorative
    ComponentScores parts;        // Component scores
    ControlScales controls;        // Robot control scales
    int64_t timestamp_ms = 0;      // Unix timestamp (milliseconds)
};

} // namespace edon

