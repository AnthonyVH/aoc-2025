add_subdirectory(common)

# Add directory for each day automatically.
file(GLOB DIRS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" day_*)
foreach(DIR ${DIRS})
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${DIR}")
    add_subdirectory(${DIR})

    # Keep track of all available days.
    string(SUBSTRING "${DIR}" 4 -1 DAY_NUMBER)
    list(APPEND DAYS ${DAY_NUMBER})
  endif()
endforeach()

# Build an executable that runs all days.
add_day_main(all_days "${DAYS}")
add_day_benchmark(all_days-benchmark "${DAYS}")
