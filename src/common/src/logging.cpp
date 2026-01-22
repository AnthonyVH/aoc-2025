#include "aoc25/logging.hpp"

#include <spdlog/cfg/argv.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

namespace aoc25 {

  void setup_logging(int argc, char ** argv) {
    // Setup default logging level.
    spdlog::set_level(static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));

    // Try and set logging from env and command line args (in that order of preference).
    spdlog::cfg::load_env_levels();
    spdlog::cfg::load_argv_levels(argc, argv);
  }

}  // namespace aoc25
