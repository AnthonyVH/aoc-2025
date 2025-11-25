configure BUILD_TYPE="Debug":
  @echo "Configuring build for type: {{BUILD_TYPE}}"
  mkdir -p build/{{BUILD_TYPE}}
  cmake -B build/{{BUILD_TYPE}} \
    -G Ninja \
    -DCMAKE_BUILD_TYPE={{BUILD_TYPE}} \
    -DCMAKE_COLOR_DIAGNOSTICS=ON \
    .

build BUILD_TYPE="Debug" TARGET="all":
  @echo "Building target \"{{TARGET}}\" for \"{{BUILD_TYPE}}\" build"
  cmake --build build/{{BUILD_TYPE}} --target {{TARGET}}

run BUILD_TYPE="Debug" TARGET="aoc25-all_days" *EXE_ARGS: (build BUILD_TYPE TARGET)
  @# Find the executable (assuming target name matches executable name).
  @echo "Running target \"{{TARGET}}\" for \"{{BUILD_TYPE}}\" build"; \
   find build/{{BUILD_TYPE}}/ -type f -name "{{TARGET}}" -executable -exec {} {{EXE_ARGS}} \;

debug BUILD_TYPE TARGET: (build BUILD_TYPE TARGET)
  @# Find the executable (assuming target name matches executable name).
  @echo "Debugging target \"{{TARGET}}\" for \"{{BUILD_TYPE}}\" build"; \
   find build/{{BUILD_TYPE}}/ -type f -name "{{TARGET}}" -executable -exec gdb -ex r {} \;

test BUILD_TYPE="Debug": (build BUILD_TYPE "all")
  @echo "Testing for \"{{BUILD_TYPE}}\" build"
  ctest --test-dir build/{{BUILD_TYPE}} --output-on-failure
