add_day(12 DAY_12_LIB)

target_link_libraries(${DAY_12_LIB}
  PRIVATE
    hwy
    OpenMP::OpenMP_CXX
)
