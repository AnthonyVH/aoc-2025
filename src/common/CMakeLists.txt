add_library(aoc25-common
  STATIC
)

target_sources(aoc25-common
  PRIVATE
    "src/day.cpp"
    "src/file.cpp"
    "src/string.cpp"
    "src/system.cpp"
)

target_include_directories(aoc25-common
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(aoc25-common
  PUBLIC
    fmt::fmt
    hwy
    spdlog::spdlog
)
