cmake_minimum_required(VERSION 3.28)
project(aoc-2025 LANGUAGES CXX)

# Add custom CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(AddDay)
include(CTest)
include(FetchContent)

# Enable generation of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard and compiler options
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Always build specifically for the host machine
add_compile_options("-march=native")

# Keep track whether this is a release build
set(IS_RELEASE_BUILD OFF)
if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(IS_RELEASE_BUILD ON)
endif()

# Configure LTO
set(ENABLE_LTO OFF)
if (IS_RELEASE_BUILD)
  set(ENABLE_LTO ON)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # Enable LTO
endif()

# Enable proper stack traces for RelWithDebInfo
if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  add_compile_options("-fno-omit-frame-pointer")
endif()

# Enable sanitizers in Debug builds
if (NOT IS_RELEASE_BUILD)
  set(SANITIZERS "-fsanitize=address,undefined")
  add_compile_options(${SANITIZERS} -fno-omit-frame-pointer)
  add_link_options(${SANITIZERS})
endif()

# Every warning is an error
add_compile_options(-Wall -Wextra -Werror -Wshadow)

# Use libc++ when using Clang
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-stdlib=libc++)
  add_link_options(-stdlib=libc++)
endif()

# Set default log level based on build type
if (IS_RELEASE_BUILD)
  add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)
else()
  add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
endif()

# Enable OpenMP support
find_package(OpenMP REQUIRED)

# Fetch external dependencies
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  boost-preprocessor
  SYSTEM
  GIT_REPOSITORY https://github.com/boostorg/preprocessor.git
  GIT_TAG        boost-1.89.0
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(boost-preprocessor)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Enable testing of the benchmark library." FORCE)
set(BENCHMARK_ENABLE_LTO ${ENABLE_LTO} CACHE BOOL "Enable link time optimisation of the benchmark library." FORCE)
FetchContent_Declare(
  google-benchmark
  SYSTEM
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.9.4
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(google-benchmark)

set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "Include contrib/" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "Install library" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "Enable HWY tests" FORCE)
FetchContent_Declare(
  google-highway
  SYSTEM
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG        1.3.0
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(google-highway)

# Define HWY_COMPILE_ONLY_STATIC for any target that uses Google Highway. This will cause it to
# only compile its code for static dispatch (i.e. for the host machine's instruction set). Which
# is fine, since we only intend to run these executables on the local machine. Note that this
# only works properly because we always set -march=native above.
target_compile_definitions(hwy PUBLIC HWY_COMPILE_ONLY_STATIC)

FetchContent_Declare(
  eigen
  SYSTEM
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        5.0.1
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  fmt
  SYSTEM
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        12.1.0
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(
  kokkos-mdspan
  SYSTEM
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG        stable
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(kokkos-mdspan)

FetchContent_Declare(
  magic_enum
  SYSTEM
  GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
  GIT_TAG        v0.9.7
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(magic_enum)

set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "Use external fmt library" FORCE)
FetchContent_Declare(
  spdlog
  SYSTEM
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.16.0
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(spdlog)

# Add sources
add_subdirectory(src)
