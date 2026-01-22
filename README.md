# :christmas_tree: Advent of Code 2025 :christmas_tree:

## :clipboard: Goals

I set myself a few goals to keep things interesting/challenging:

  - [x] No discussion of or reading about solutions before I consider this finished.
  - [x] Solve all problems.
  - [x] Solve all problems in under 12 ms total.
  - [x] Solve all problems in under 1 ms each.
  - [x] Solve all of a day's problems in under 1 ms combined.
  - [ ] Solve all problems in under 1 ms combined.

Clearly the main goal was: make things as fast as possible. Unfortunately I didn't manage the "everything in under 1 ms" challenge. I got very close though, with a combined runtime of 1.10 ms for all problems.

> [!NOTE]
> None of the implementations are tuned to my inputs. I.e. any input conforming to what is given as an example on the Advent of Code's website should work. Of course I can't test that, since I don't have access to other input files.

## :rocket: Benchmarks

The runtimes below were measured using [Google Benchmark](https://github.com/google/benchmark). The setup used for these results was:

- Intel Core i5-12400
- 64 GiB PC3200 DDR4
- Ubuntu 24.04 LTS
- GCC 14.2.0
- Executed with `nice --adjustment=-10`

Compilation was done in CMake's `Release` mode, with profile-guided optimization and link-time optimization.

> [!IMPORTANT]
> The measured times include parsing the input text, and solving the problem. They do not include reading the input text from a file or standard input.

```
╭────────────────────────────────────────────────────────────╮
│ Benchmark                Time             CPU   Iterations │
╰────────────────────────────────────────────────────────────╯
  day 01 - part 1       14.1 us         14.1 us        49014
  day 01 - part 2       17.9 us         17.9 us        39083
  day 02 - part 1      0.600 us        0.600 us      1162240
  day 02 - part 2       1.11 us         1.11 us       628806
  day 03 - part 1       2.02 us         2.02 us       251657
  day 03 - part 2       4.17 us         4.17 us       173261
  day 04 - part 1       2.27 us         2.27 us       308822
  day 04 - part 2        104 us          103 us         6749
  day 05 - part 1       14.5 us         14.5 us        48263
  day 05 - part 2       4.92 us         4.92 us       142410
  day 06 - part 1       5.49 us         5.49 us       124417
  day 06 - part 2       5.78 us         5.78 us       119230
  day 07 - part 1      0.861 us        0.861 us       815084
  day 07 - part 2       5.40 us         5.39 us       129339
  day 08 - part 1        278 us          277 us         2527
  day 08 - part 2        175 us          175 us         4004
  day 09 - part 1       5.77 us         5.77 us       120859
  day 09 - part 2       84.9 us         84.7 us         8255
  day 10 - part 1       73.1 us         73.0 us         9579
  day 10 - part 2        274 us          274 us         2564
  day 11 - part 1       5.35 us         5.34 us       130514
  day 11 - part 2       10.3 us         10.3 us        67973
  day 12 - part 1       5.73 us         5.73 us       121481
  ──────────────────────────────────────────────────────────
  Total                 1095 us         1093 us
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
