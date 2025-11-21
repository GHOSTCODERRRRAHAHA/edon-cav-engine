cmake_minimum_required(VERSION 3.16)
project(edon-sdk-cpp VERSION 1.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Protobuf generation
set(PROTO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../integrations/grpc/edon_grpc_service")
set(PROTO_FILE "${PROTO_PATH}/edon.proto")

# Generate protobuf and gRPC code
set(PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/edon.pb.cc")
set(PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/edon.pb.h")
set(GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/edon.grpc.pb.cc")
set(GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/edon.grpc.pb.h")

add_custom_command(
    OUTPUT "${PROTO_SRCS}" "${PROTO_HDRS}" "${GRPC_SRCS}" "${GRPC_HDRS}"
    COMMAND protobuf::protoc
    ARGS --proto_path="${PROTO_PATH}"
         --cpp_out="${CMAKE_CURRENT_BINARY_DIR}"
         --grpc_out="${CMAKE_CURRENT_BINARY_DIR}"
         --plugin=protoc-gen-grpc="${gRPC_CPP_PLUGIN_EXECUTABLE}"
         "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
    COMMENT "Generating protobuf and gRPC code from edon.proto"
)

# Library sources
set(LIB_SRCS
    src/edon.cpp
    src/transport_grpc.cpp
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

# Create library
add_library(edon_sdk STATIC ${LIB_SRCS})

target_link_libraries(edon_sdk
    PRIVATE
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
)

target_include_directories(edon_sdk
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

# Example executable (robot example)
add_executable(edon_robot_example examples/example.cpp)
target_link_libraries(edon_robot_example PRIVATE edon_sdk)
set_target_properties(edon_robot_example PROPERTIES
    OUTPUT_NAME "edon_robot_example"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS edon_sdk
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/edon
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

