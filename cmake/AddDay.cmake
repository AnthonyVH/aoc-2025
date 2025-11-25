# Cache path to this module, so we can reference files relative to it in
# functions defined here.
set(ADD_DAY_MODULE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function (add_day_exe_from_template EXE_SUFFIX TEMPLATE_FILE DAY_NUMBERS EXE_TARGET_OUT)
  set(EXE_TARGET "aoc25-${EXE_SUFFIX}")

  # Return executable target to caller.
  set(${EXE_TARGET_OUT} "${EXE_TARGET}" PARENT_SCOPE)

  add_executable(${EXE_TARGET}
    "${ADD_DAY_MODULE_DIR}/templates/${TEMPLATE_FILE}"
  )

  set_target_properties(${EXE_TARGET}
    PROPERTIES
      OUTPUT_NAME "aoc25-${EXE_SUFFIX}"
  )

  target_link_libraries(${EXE_TARGET}
    PRIVATE
      aoc25-common
      fmt::fmt
      spdlog::spdlog
  )

  target_compile_definitions(${EXE_TARGET}
    PRIVATE
      "INPUT_DIR=inputs" # Relative to root
  )

  # Allow compiler to find code for each day.
  # Note that we can't use a macro to expand a list and then
  # include multiple files based on that list. Hence we need
  # to generate an "include header" here.
  set(GENERATED_HEADER_CONTENT "")
  foreach(DAY_NUMBER ${DAY_NUMBERS})
    string(APPEND GENERATED_HEADER_CONTENT "#include \"aoc25/day_${DAY_NUMBER}.hpp\"\n")
  endforeach()

  set(GENERATED_HEADER_PATH "${CMAKE_CURRENT_BINARY_DIR}/generated_includes.hpp")
  file(WRITE ${GENERATED_HEADER_PATH} "${GENERATED_HEADER_CONTENT}")
  target_include_directories(${EXE_TARGET}
    PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR}
  )

  target_compile_definitions(${EXE_TARGET}
    PRIVATE
      "GENERATED_HEADER_FILE=\"generated_includes.hpp\""
  )

  # Create comma-separated list of days, force into decimal form first by removing leading zeros.
  string(REGEX REPLACE "0*([0-9]+)" "\\1" DAY_NUMBERS_CPP "${DAY_NUMBERS}")
  string(JOIN ", " DAY_NUMBERS_CPP ${DAY_NUMBERS_CPP})
  target_compile_definitions(${EXE_TARGET}
    PRIVATE
      "DAY_NUMBERS=${DAY_NUMBERS_CPP}"
  )

  foreach(DAY_NUMBER ${DAY_NUMBERS})
    target_link_libraries(${EXE_TARGET}
      PRIVATE
        aoc25-day_${DAY_NUMBER}-lib
    )
  endforeach()
endfunction()

function (add_day_main EXE_SUFFIX DAY_NUMBERS)
  add_day_exe_from_template(${EXE_SUFFIX} "main.cpp" "${DAY_NUMBERS}" IGNORE)
endfunction()

function (add_day_benchmark EXE_SUFFIX DAY_NUMBERS)
  add_day_exe_from_template(${EXE_SUFFIX} "benchmark.cpp" "${DAY_NUMBERS}" TARGET)

  target_link_libraries(${TARGET}
    PRIVATE
      benchmark::benchmark
  )
endfunction()

function (add_day_verify EXE_SUFFIX DAY_NUMBERS)
  add_day_exe_from_template(${EXE_SUFFIX} "verify.cpp" "${DAY_NUMBERS}" TARGET)

  add_test(
    NAME ${TARGET}
    COMMAND ${TARGET}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
endfunction()

function(add_day DAY_NUMBER LIB_NAME_OUT)
  set(LIB_NAME "aoc25-day_${DAY_NUMBER}-lib")

  # Return library target to caller.
  set(${LIB_NAME_OUT} "${LIB_NAME}" PARENT_SCOPE)

  add_library(${LIB_NAME} STATIC)

  target_sources(${LIB_NAME}
    PRIVATE
      "src/day_${DAY_NUMBER}.cpp"
  )

  target_include_directories(${LIB_NAME}
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )

  target_link_libraries(${LIB_NAME}
    PUBLIC
      aoc25-common
    PRIVATE # Some common dependencies
      fmt::fmt
      spdlog::spdlog
  )

  add_day_main(day_${DAY_NUMBER} ${DAY_NUMBER})
  add_day_benchmark(day_${DAY_NUMBER}-benchmark ${DAY_NUMBER})
  add_day_verify(day_${DAY_NUMBER}-verify ${DAY_NUMBER})
endfunction()
