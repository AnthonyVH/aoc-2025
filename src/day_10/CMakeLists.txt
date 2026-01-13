add_day(10 DAY_10_LIB)

target_link_libraries(${DAY_10_LIB}
  PRIVATE
    eigen
    hwy
    OpenMP::OpenMP_CXX
)
