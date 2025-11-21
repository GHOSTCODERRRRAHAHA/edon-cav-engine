/**
 * @file example.cpp
 * @brief Example usage of EDON C++ SDK - Robot Integration
 */

#include "edon/edon.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

int main() {
    try {
        std::cout << "EDON C++ SDK - Robot Example" << std::endl;
        std::cout << "============================" << std::endl << std::endl;
        
        // Create client
        edon::EdonClient client("localhost", 50051);
        
        // Health check
        if (!client.health()) {
            std::cerr << "ERROR: gRPC server not available" << std::endl;
            return 1;
        }
        std::cout << "[OK] Connected to EDON gRPC server" << std::endl << std::endl;
        
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
        std::cout << "Computing CAV..." << std::endl;
        edon::CAVResponse response = client.computeCAV(window);
        
        std::cout << "State: " << response.state << std::endl;
        std::cout << "CAV Smooth: " << response.cav_smooth << std::endl;
        std::cout << "P-Stress: " << response.parts.p_stress << std::endl;
        std::cout << "Controls - Speed: " << response.controls.speed 
                  << ", Torque: " << response.controls.torque 
                  << ", Safety: " << response.controls.safety << std::endl;
        
        // Classify
        std::string state = client.classify(window);
        std::cout << "\nClassified state: " << state << std::endl;
        
        // Robot control loop example
        std::cout << "\nRobot Control Loop (5 iterations):" << std::endl;
        std::cout << "-----------------------------------" << std::endl;
        
        for (int i = 0; i < 5; i++) {
            response = client.computeCAV(window);
            
            // Map state to control scales
            float speed, torque, safety;
            if (response.state == "restorative") {
                speed = 0.7f; torque = 0.7f; safety = 0.95f;
            } else if (response.state == "balanced") {
                speed = 1.0f; torque = 1.0f; safety = 0.85f;
            } else if (response.state == "focus") {
                speed = 1.2f; torque = 1.1f; safety = 0.8f;
            } else { // overload
                speed = 0.4f; torque = 0.4f; safety = 1.0f;
            }
            
            std::cout << "[STEP " << i << "] State=" << response.state 
                      << ", P-Stress=" << response.parts.p_stress
                      << " -> Speed=" << speed 
                      << ", Torque=" << torque 
                      << ", Safety=" << safety << std::endl;
            
            // Simulate robot applying controls
            // apply_scales_to_controllers(speed, torque, safety);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "\n[OK] Example complete!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

