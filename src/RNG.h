#ifndef RNG_H
#define RNG_H

#include <cmath>
#include <cstddef>
#include <random>

namespace nn {

/// Process-wide Mersenne Twister; a single instance is shared by all
/// translation units. Seeded once with DEFAULT_SEED for reproducible runs.
inline std::mt19937 &rng()
{
   static std::mt19937 generator{1};
   return generator;
}

inline void seed_rng(std::size_t seed)
{
   rng().seed(static_cast<std::mt19937::result_type>(seed));
}

/// Glorot/Xavier uniform init, bound = +/- sqrt(6 / (fan_in + fan_out)).
inline double xavier_init(std::size_t fan_in, std::size_t fan_out)
{
   const std::size_t denom = fan_in + fan_out;
   if (denom == 0) {
      return 0.0;
   }
   const double limit = std::sqrt(6.0 / static_cast<double>(denom));
   std::uniform_real_distribution<double> dist{-limit, limit};
   return dist(rng());
}

/// He init for the ReLU activation family, bound = +/- sqrt(6 / fan_in).
inline double he_init(std::size_t fan_in)
{
   return xavier_init(fan_in, 0);
}

} // namespace nn

#endif