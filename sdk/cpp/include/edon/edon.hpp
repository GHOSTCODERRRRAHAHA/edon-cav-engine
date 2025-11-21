/**
 * @file edon.hpp
 * @brief Main EDON C++ SDK header
 */

#pragma once

#include <string>
#include <memory>
#include <functional>
#include <vector>
#include "models.hpp"
#include "transport.hpp"

namespace edon {

/**
 * @brief Main EDON client class
 */
class EdonClient {
public:
    /**
     * @brief Constructor
     * @param host gRPC server host (default: "localhost")
     * @param port gRPC server port (default: 50051)
     */
    explicit EdonClient(const std::string& host = "localhost", int port = 50051);
    
    /**
     * @brief Destructor
     */
    ~EdonClient();
    
    /**
     * @brief Compute CAV from sensor window (synchronous)
     * @param window Sensor window data
     * @return CAV response
     */
    CAVResponse computeCAV(const SensorWindow& window);
    
    /**
     * @brief Classify state from sensor window (convenience method)
     * @param window Sensor window data
     * @return State string: "overload", "balanced", "focus", or "restorative"
     */
    std::string classify(const SensorWindow& window);
    
    /**
     * @brief Stream CAV updates (asynchronous callback)
     * @param window Sensor window data
     * @param callback Function called for each update
     * @param stream_mode If true, server pushes updates continuously
     */
    void stream(
        const SensorWindow& window,
        std::function<void(const CAVResponse&)> callback,
        bool stream_mode = true
    );
    
    /**
     * @brief Check service health
     * @return True if service is healthy
     */
    bool health();

private:
    std::unique_ptr<Transport> transport_;
};

} // namespace edon

