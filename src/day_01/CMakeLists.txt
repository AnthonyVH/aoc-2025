add_day(01 DAY_01_LIB)

target_sources(${DAY_01_LIB}
  PRIVATE
    "src/day_01-hwy.cpp"
)

target_link_libraries(${DAY_01_LIB}
  PRIVATE
    hwy
)
