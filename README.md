# :christmas_tree: Advent of Code 2025 :christmas_tree:

## :clipboard: Goals

I set myself a few goals to keep things interesting/challenging:

  - [x] No discussion of or reading about solutions before I consider this finished.
  - [x] Solve all problems.
  - [x] Solve all problems in under 12 ms total.
  - [x] Solve all problems in under 1 ms each.
  - [x] Solve all of a day's problems in under 1 ms combined.
  - [ ] Solve all problems in under 1 ms combined.

Clearly the main goal was: make things as fast as possible. Unfortunately I didn't manage the "everything in under 1 ms" challenge. I got very close though, with a combined runtime of 1.29 ms for all problems.

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
  day 01 - part 1       15.1 us         15.1 us        49139
  day 01 - part 2       17.8 us         17.8 us        39063
  day 02 - part 1      0.614 us        0.614 us      1187483
  day 02 - part 2       1.12 us         1.12 us       625569
  day 03 - part 1       2.56 us         2.56 us       274452
  day 03 - part 2       4.08 us         4.07 us       174174
  day 04 - part 1       1.85 us         1.85 us       265728
  day 04 - part 2       90.7 us         90.7 us         6797
  day 05 - part 1       10.8 us         10.8 us        48181
  day 05 - part 2       3.28 us         3.28 us       142673
  day 06 - part 1       5.51 us         5.51 us       121176
  day 06 - part 2       5.87 us         5.87 us       120232
  day 07 - part 1      0.818 us        0.818 us       815155
  day 07 - part 2       5.09 us         5.09 us       129519
  day 08 - part 1        371 us          371 us         1742
  day 08 - part 2        307 us          307 us         2140
  day 09 - part 1       5.48 us         5.48 us       121624
  day 09 - part 2       74.2 us         74.2 us         9441
  day 10 - part 1       63.9 us         63.8 us        10296
  day 10 - part 2        283 us          281 us         2610
  day 11 - part 1       4.88 us         4.87 us       137230
  day 11 - part 2       9.86 us         9.86 us        72798
  day 12 - part 1       5.62 us         5.62 us       121303
  ──────────────────────────────────────────────────────────
  Total                 1290 us         1288 us
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
