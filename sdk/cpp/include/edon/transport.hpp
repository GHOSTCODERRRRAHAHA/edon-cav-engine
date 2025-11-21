/**
 * @file transport.hpp
 * @brief Transport layer abstraction
 */

#pragma once

#include "models.hpp"
#include <functional>
#include <memory>

namespace edon {

/**
 * @brief Abstract transport interface
 */
class Transport {
public:
    virtual ~Transport() = default;
    
    /**
     * @brief Compute CAV synchronously
     */
    virtual CAVResponse computeCAV(const SensorWindow& window) = 0;
    
    /**
     * @brief Stream CAV updates
     */
    virtual void stream(
        const SensorWindow& window,
        std::function<void(const CAVResponse&)> callback,
        bool stream_mode = true
    ) = 0;
    
    /**
     * @brief Check health
     */
    virtual bool health() = 0;
};

/**
 * @brief gRPC transport implementation
 */
class GRPCTransport : public Transport {
public:
    /**
     * @brief Constructor
     * @param host gRPC server host
     * @param port gRPC server port
     */
    GRPCTransport(const std::string& host, int port);
    
    ~GRPCTransport();
    
    CAVResponse computeCAV(const SensorWindow& window) override;
    
    void stream(
        const SensorWindow& window,
        std::function<void(const CAVResponse&)> callback,
        bool stream_mode = true
    ) override;
    
    bool health() override;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace edon

