# :christmas_tree: Advent of Code 2025 :christmas_tree:

## :clipboard: Goals

I set myself a few goals to keep things interesting/challenging:

  - [x] No discussion of or reading about solutions before I consider this finished.
  - [x] Solve all problems.
  - [x] Solve all problems in under 12 ms total.
  - [x] Solve all problems in under 1 ms each.
  - [x] Solve all of a day's problems in under 1 ms combined.
  - [x] Solve all problems in under 1 ms combined.

Clearly the main goal was: make things as fast as possible. The total runtime is 897 μs :tada:.

I'm sure there's room for improvement in many places, but since I reached all my goals, and already spend _way_ too much time on this, this will have to do for now.

> [!NOTE]
> None of the implementations are tuned to my inputs. I.e. any input conforming to what is given as an example on the Advent of Code's website should work. Of course I can't test that, since I don't have access to other input files.

## :rocket: Benchmarks

The runtimes below were measured using [Google Benchmark](https://github.com/google/benchmark). The setup used for these results was:

- Intel Core i5-12400
- 64 GiB PC3200 DDR4
- Ubuntu 24.04 LTS
- GCC 14.2.0
- Executed with `nice --adjustment=-10`
- Using [`mimalloc 2.1.2`](https://github.com/microsoft/mimalloc) instead of `malloc`

Compilation was done in CMake's `Release` mode, with link-time optimization (but profile-guided optimization disabled, because it actually resulted in slower runtimes).

The runtimes were measured for each day individually. GCC clearly had an easier time optimizing multiple small binaries instead of a single big one that computed the solutions for all days.

> [!IMPORTANT]
> The measured times include parsing the input text, and solving the problem. They do not include reading the input text from a file or standard input.

```
╭────────────────────────────────────────────────────────────╮
│ Benchmark                Time             CPU   Iterations │
╰────────────────────────────────────────────────────────────╯
  day 01 - part 1       14.2 us         14.2 us        48453
  day 01 - part 2       17.8 us         17.8 us        39272
  day 02 - part 1      0.625 us        0.625 us      1072479
  day 02 - part 2       1.12 us         1.12 us       626535
  day 03 - part 1       1.92 us         1.92 us       365210
  day 03 - part 2       4.80 us         4.80 us       145468
  day 04 - part 1       1.85 us         1.85 us       375385
  day 04 - part 2       90.9 us         90.8 us         7730
  day 05 - part 1       15.3 us         15.3 us        43774
  day 05 - part 2       3.29 us         3.29 us       212310
  day 06 - part 1       5.22 us         5.22 us       132612
  day 06 - part 2       5.29 us         5.28 us       132633
  day 07 - part 1      0.754 us        0.753 us       898532
  day 07 - part 2       7.31 us         7.31 us        96019
  day 08 - part 1        262 us          262 us         2662
  day 08 - part 2        174 us          174 us         4020
  day 09 - part 1       5.49 us         5.48 us       121090
  day 09 - part 2       52.3 us         52.3 us        13392
  day 10 - part 1       19.2 us         19.2 us        36649
  day 10 - part 2        193 us          193 us         3614
  day 11 - part 1       5.44 us         5.42 us       121944
  day 11 - part 2       9.77 us         9.77 us        70804
  day 12 - part 1       5.59 us         5.59 us       124677
───────────────────────────────────────────────────────────────
  Total                  897 us          897 us
```

## :hammer: Building

To build everything, the following is required at minimum:

- CMake >= 3.28
- GCC >= 14.2.0

For convenience, there's also a `justfile` included (which requires `just` to be installed). Just run `just` to see the available commands.

Here are a few common uses:

- Configure `Release` builds: `just configure Release`
- Build & run the benchmark: `just run aoc25-all_days-benchmark`
- Build & verify day 3's solutions: `just run aoc25-day_03-verify`

## :page_facing_up: Input data

All binaries expect input data to be available in a folder `inputs` below the current working directory. Since the Advent of Code website explicitly requests not to share one's input data, you'll have to provide your own.

The files are expected to have the following naming format (in [PCRE2 syntax](https://regex101.com/)): `day_\d{2}-part_\d+\.txt`. So, these are all valid filenames:

- `day_01-part_1.txt`
- `day_03-part_2.txt`

This is not a valid filename: `day_1-part_1.txt`, because there's only one digit for the day.

All binaries will automatically scan for files matching this pattern, and run the solver for each of them.

The `...-verify` binaries will also search for example files (pattern: `day_\d{2}-part_\d+-example_\d+\.txt`). For all files it finds, it expects a matching `...-solution.txt` file to exist (i.e. `day_\d{2}-part_\d+(?:-example_\d+)?-solution.txt`). These solution files are used to check the output of the solver.

## :pencil2: Logging

All output is done via [`spdlog`](https://github.com/gabime/spdlog), which means the default log levels can be overriden by setting the `SPDLOG_LEVEL` enviroment variable to one of [`trace`, `debug`, `info`, `warning`, or `error`].

Note that depending on the build type all log calls equal to and below a certain level are completely removed during compilation (i.e. not just their output disabled). For `Debug` builds all `trace` calls are removed, and for `Release`/`RelWithDebInfo` builds all `debug` (and lower) calls are removed. So e.g. setting `SPDLOG_LEVEL=debug` won't result in any `debug`-level logs in a `Release` build.
