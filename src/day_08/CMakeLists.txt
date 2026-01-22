add_day(08 DAY_08_LIB)

target_link_libraries(${DAY_08_LIB}
  PRIVATE
    hwy
    OpenMP::OpenMP_CXX
)
