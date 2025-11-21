/**
 * @file edon.cpp
 * @brief EDON client implementation
 */

#include "edon/edon.hpp"
#include "edon/transport.hpp"
#include <stdexcept>

namespace edon {

EdonClient::EdonClient(const std::string& host, int port)
    : transport_(std::make_unique<GRPCTransport>(host, port))
{
}

EdonClient::~EdonClient() = default;

CAVResponse EdonClient::computeCAV(const SensorWindow& window) {
    return transport_->computeCAV(window);
}

std::string EdonClient::classify(const SensorWindow& window) {
    CAVResponse response = computeCAV(window);
    return response.state;
}

void EdonClient::stream(
    const SensorWindow& window,
    std::function<void(const CAVResponse&)> callback,
    bool stream_mode
) {
    transport_->stream(window, callback, stream_mode);
}

bool EdonClient::health() {
    return transport_->health();
}

} // namespace edon

