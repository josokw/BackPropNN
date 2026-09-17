#ifndef NNCONFIG_H
#define NNCONFIG_H

#include <cstddef>

const std::size_t MAX_ITERATIONS = 1'000'000;
const double MIN_RECENT_AVERAGE_ERROR = 0.03;

///< Default learning rate when the config file omits ETA.
inline constexpr double DEFAULT_ETA = 0.15;
///< Default momentum when the config file omits ALPHA.
inline constexpr double DEFAULT_ALPHA = 0.5;

[[nodiscard]] inline bool do_show(std::size_t iteration, double average_error)
{
   return (iteration % 1000 == 0) or
          (average_error < 1.04 * MIN_RECENT_AVERAGE_ERROR);
}

#endif
