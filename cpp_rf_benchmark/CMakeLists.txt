cmake_minimum_required(VERSION 3.15)
project(RandomForestBenchmark VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages
find_package(Threads REQUIRED)
#find_package(nlohmann_json 3.2.0 REQUIRED)

# Include nlohmann/json (header-only)
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Source files
set(SOURCES
    src/main.cpp
    src/random_forest.cpp
    src/decision_tree.cpp
    src/data_loader.cpp
    src/cpu_monitor.cpp
    src/benchmark.cpp
    src/report_generator.cpp
    src/utils.cpp
)

set(HEADERS
    include/random_forest.h
    include/decision_tree.h
    include/data_loader.h
    include/cpu_monitor.h
    include/benchmark.h
    include/report_generator.h
    include/utils.h
    include/types.h
)

# Create executable
add_executable(rf_benchmark ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(rf_benchmark PRIVATE include)

# Link libraries
target_link_libraries(rf_benchmark PRIVATE 
    Threads::Threads
    nlohmann_json::nlohmann_json
)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(rf_benchmark PRIVATE 
        -Wall -Wextra -O3 -march=native -pthread
    )
endif()

# Install targets
install(TARGETS rf_benchmark DESTINATION bin)
